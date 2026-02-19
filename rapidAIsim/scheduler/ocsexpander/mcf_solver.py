import random
from collections import defaultdict
import numpy as np
from ortools.graph.python import min_cost_flow
import time
import multiprocessing as mp


class MCFSolver:
    """
    该类用于迭代求解ocs-expander中minimal-rewiring的问题。
    """
    def __init__(self, pod_num, spine_num_per_pod, oxc_list, original_oxc_to_each_spine_num, c_ijt, u_ijkt,
                 port_per_spine, tradition_mcf = False):
        # print("debug oxc_list")
        # print(oxc_list)
        # print("debug c_ijt")
        # print(c_ijt)
        self.oxc_list = [oxc_list[i] for i in range(len(oxc_list)) if i % 2 == 0]
        self.original_oxc_to_each_spine_num = original_oxc_to_each_spine_num
        self.pod_num = pod_num
        self.oxc_num = len(oxc_list)
        self.spine_num_per_pod = spine_num_per_pod
        self.port_per_spine = port_per_spine
        self.oxc_physical_group_size = self.port_per_spine // 2
        self.oxc_physical_group_num = self.spine_num_per_pod
        # self.pool = mp.Pool()
        self.c_ijt = c_ijt
        self.u_ijkt = u_ijkt
        self.spine_connected_oxc_list_map = defaultdict(list)

    def solve(self, need_return_u = False):
        oxc_group_0 = np.sort(np.array([self.oxc_list[i + t * self.oxc_physical_group_size]
                                        for i in range(self.oxc_physical_group_size) if i % 2 == 0
                                        for t in range(self.oxc_physical_group_num)]))
        oxc_group_1 = np.sort(np.array([self.oxc_list[i + t * self.oxc_physical_group_size]
                                        for i in range(self.oxc_physical_group_size) if i % 2 == 1
                                        for t in range(self.oxc_physical_group_num)]))
        oxc_groups = [oxc_group_0, oxc_group_1]

        for t in range(self.spine_num_per_pod):
            for logical_local_oxc_id in range(self.oxc_physical_group_size):
                logical_global_oxc_id = logical_local_oxc_id + self.oxc_physical_group_size * t
                self.spine_connected_oxc_list_map[t].append(2 * logical_global_oxc_id)
        tmp_x_ijkt = self.solve_merge_and_decomp(oxc_groups, self.c_ijt, self.u_ijkt, 0)
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.oxc_physical_group_size * 2, self.spine_num_per_pod),
                          dtype=int)
        for k in range(0, self.oxc_physical_group_size, 1):
            x_ijkt[:, :, 2 * k, :] = tmp_x_ijkt[:, :, k, :]
            
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    assert np.sum(x_ijkt[i, j, :, t])>=self.c_ijt[i,j,t]

        # self.check_tmp_MCF(self.A_ij, original_oxc_to_each_spine_num)
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for k in range(0, self.oxc_physical_group_size * 2, 2):
                    x_ijkt[j, i, k + 1, :] = x_ijkt[i, j, k, :]

        # self.check_MCF(c_ij, original_oxc_to_each_spine_num)
        if need_return_u:
            return x_ijkt, tmp_x_ijkt
        else:
            return x_ijkt

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        try:
            del self_dict['pool']
        except KeyError:
            pass
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def solve_merge_and_decomp(self, oxc_groups, c_ijt, u_ijkt, cur_stage_num, cijt_v=None):
        # print("debug oxc_group_map",oxc_group_map)
        pod_num = self.pod_num
        spine_num_per_pod = self.spine_num_per_pod
        original_oxc_to_each_spine_num = self.original_oxc_to_each_spine_num

        oxc_group_0, oxc_group_1 = oxc_groups
        oxc_to_group_map = {k: 0 for k in oxc_group_0}
        oxc_to_group_map.update({k: 1 for k in oxc_group_1})
        cur_oxc_list = []
        cur_oxc_list.extend(oxc_group_0)
        cur_oxc_list.extend(oxc_group_1)

        if cijt_v is not None:
            c_ijt = cijt_v

        # 注意，此时的k只取0或者1，是oxc group的id
        u_star_ijkt = np.zeros((pod_num, pod_num, 2, spine_num_per_pod), dtype=int)
        u_star_ijkt[:, :, 0, :] = np.sum(u_ijkt[:, :, np.mod(oxc_group_0, self.oxc_physical_group_size), :], axis=2)
        u_star_ijkt[:, :, 1, :] = np.sum(u_ijkt[:, :, np.mod(oxc_group_1, self.oxc_physical_group_size), :], axis=2)

        # 根据c_ijt生成logical_a_ik，logical_b_jk
        logical_a_it = np.sum(c_ijt, axis=1)
        logical_b_jt = np.sum(c_ijt, axis=0)

        random.seed(0)
        # 根据oxc分组生成logical_spine_oxc_link和logical_spine_oxc_link
        logical_spine_oxc_link = np.zeros((pod_num, spine_num_per_pod, 2), dtype=int)
        for i in range(pod_num):
            for t in range(spine_num_per_pod):
                for k in cur_oxc_list:
                    if k in self.spine_connected_oxc_list_map[t]:
                        logical_spine_oxc_link[i, t, oxc_to_group_map[k]] += original_oxc_to_each_spine_num

        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.oxc_physical_group_size, self.spine_num_per_pod), dtype=int)

        # 调用求解器，求出oxc合并后的解
        x_star_ijkf = self.solve_single_MCF(c_ijt, u_star_ijkt, logical_a_it, logical_b_jt, logical_spine_oxc_link)

        if (len(oxc_group_0) <= 1 * self.oxc_physical_group_num
                and len(oxc_group_1) <= 1 * self.oxc_physical_group_num):
            for t in range(spine_num_per_pod):
                for k in set(oxc_group_0).intersection(set(self.spine_connected_oxc_list_map[t])):
                    x_ijkt[:, :, (k // 2) % self.oxc_physical_group_size, t] = x_star_ijkf[:, :, 0, t]
                    # print(t, k % self.oxc_physical_group_size)
                for k in set(oxc_group_1).intersection(set(self.spine_connected_oxc_list_map[t])):
                    x_ijkt[:, :, (k // 2) % self.oxc_physical_group_size, t] = x_star_ijkf[:, :, 1, t]
                    # print(t, k % self.oxc_physical_group_size)
            return x_ijkt

        # 拆分子图
        # 设定子图的cij需求
        c_sub_ijt = [x_star_ijkf[:, :, 0, :].copy(), x_star_ijkf[:, :, 1, :].copy()]

        # 设定子图的uijk
        u_sub_ijkt_0 = np.zeros_like(u_ijkt)
        u_sub_ijkt_1 = np.zeros_like(u_ijkt)

        u_ijkt_copy = u_ijkt.copy()
        u_sub_ijkt_0[:, :, np.mod(oxc_group_0, self.oxc_physical_group_size), :] = \
            u_ijkt_copy[:, :, np.mod(oxc_group_0, self.oxc_physical_group_size), :].copy()
        u_sub_ijkt_1[:, :, np.mod(oxc_group_1, self.oxc_physical_group_size), :] = \
            u_ijkt_copy[:, :, np.mod(oxc_group_1, self.oxc_physical_group_size), :].copy()
        u_sub_ijkt = [u_sub_ijkt_0, u_sub_ijkt_1]
        # 设定新的oxc_group_map
        oxc_group0_size = len(oxc_group_0) // self.oxc_physical_group_num
        oxc_group1_size = len(oxc_group_1) // self.oxc_physical_group_num
        sub_oxc_groups = [[np.sort(np.array([oxc_group_0[i + t * oxc_group0_size]
                                             for i in range(oxc_group0_size)
                                             for t in range(self.oxc_physical_group_num) if i % 2 == 0])),
                           np.sort(np.array([oxc_group_0[i + t * oxc_group0_size]
                                             for i in range(oxc_group0_size)
                                             for t in range(self.oxc_physical_group_num) if i % 2 == 1]))],
                          [np.sort(np.array([oxc_group_1[i + t * oxc_group1_size]
                                             for i in range(oxc_group1_size)
                                             for t in range(self.oxc_physical_group_num) if i % 2 == 0])),
                           np.sort(np.array([oxc_group_1[i + t * oxc_group1_size]
                                             for i in range(oxc_group1_size)
                                             for t in range(self.oxc_physical_group_num) if i % 2 == 1]))]]

        # if cur_stage_num == 0:
        #     res0 = self.pool.apply_async(self.solve_merge_and_decomp, args=(
        #         sub_oxc_groups[0], c_sub_ijt[0], u_sub_ijkt[0], cur_stage_num + 1, c_sub_ijt[0]))
        #     res1 = self.pool.apply_async(self.solve_merge_and_decomp, args=(
        #         sub_oxc_groups[1], c_sub_ijt[1], u_sub_ijkt[1], cur_stage_num + 1, c_sub_ijt[1]))
        #     sub_x_ijkt_0 = res0.get()
        #     sub_x_ijkt_1 = res1.get()
        # else:
        sub_x_ijkt_0 = self.solve_merge_and_decomp(sub_oxc_groups[0], c_sub_ijt[0], u_sub_ijkt[0],
                                                   cur_stage_num + 1, c_sub_ijt[0])
        sub_x_ijkt_1 = self.solve_merge_and_decomp(sub_oxc_groups[1], c_sub_ijt[1], u_sub_ijkt[1],
                                                   cur_stage_num + 1, c_sub_ijt[1])
        return sub_x_ijkt_0 + sub_x_ijkt_1

    def solve_single_MCF(self, c_ijt, u_star_ijkt, logical_a_it, logical_b_jt, logical_spine_oxc_link):
        pod_num = self.pod_num
        spine_num_per_pod = self.spine_num_per_pod

        arc_capacity_k0_map = {}
        arc_cost_k0_map = {}
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    u_0 = u_star_ijkt[i, j, 0, t]
                    u_1 = u_star_ijkt[i, j, 1, t]
                    c = c_ijt[i, j, t]
                    if c <= 0:
                        arc_cost_k0_map[f'{i}_{j}_{t}'] = [0]
                        arc_capacity_k0_map[f'{i}_{j}_{t}'] = [0]
                    elif u_0 < c - u_1:  # 当x > c-u_1 > u_0, 斜率为2
                        # arc_capacity_k0_map[f'{i}_{j}_{t}'] = [
                        arc_capacity_k0_map[f'{i}_{j}_{t}'] = [u_0, c - u_1 - u_0, u_1]
                        arc_cost_k0_map[f'{i}_{j}_{t}'] = [-2, 0, 2]
                    elif u_0 == c - u_1:
                        arc_capacity_k0_map[f'{i}_{j}_{t}'] = [u_0, u_1]
                        arc_cost_k0_map[f'{i}_{j}_{t}'] = [-2, 2]
                    else:  # 当x > u_0 > c-u_1 , 斜率为2
                        arc_capacity_k0_map[f'{i}_{j}_{t}'] = [
                            max(0, c - u_1), u_0 - max(0, c - u_1) + min(0, c - u_0), max(0, c - u_0)]
                        arc_cost_k0_map[f'{i}_{j}_{t}'] = [-2, 0, 2]

        start_nodes = []  # 在图中，supply node为0->self.pod_num*self.spine_num_per_pod - 1
        end_nodes = []
        capacities = []
        unit_costs = []

        supplies_map = {}

        global_arc_id = 0
        i_j_t_localarc_2_globalarc_map = {}

        total_spine_nums = self.pod_num * self.spine_num_per_pod
        start_node_id_k0_start_index = total_spine_nums
        end_node_id_k0_start_index = 2 * total_spine_nums
        demand_node_id_start_index = 3 * total_spine_nums

        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    supply_node_id_k0 = self.pod_num * self.spine_num_per_pod + self.spine_num_per_pod * i + t
                    end_node_id_k0 = 2 * self.pod_num * self.spine_num_per_pod + self.spine_num_per_pod * j + t
                    for arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}_{t}'])):
                        if arc_capacity_k0_map[f'{i}_{j}_{t}'][arc_id] > 0:
                            start_nodes.append(supply_node_id_k0)
                            end_nodes.append(end_node_id_k0)
                            capacities.append(arc_capacity_k0_map[f'{i}_{j}_{t}'][arc_id])
                            unit_costs.append(arc_cost_k0_map[f'{i}_{j}_{t}'][arc_id])
                            i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{t}_{arc_id}_0'] = global_arc_id
                            global_arc_id += 1

        for i in range(self.pod_num):
            for t in range(self.spine_num_per_pod):
                a_it_node = self.spine_num_per_pod * i + t
                supply_node_id_k0 = self.spine_num_per_pod * i + t + self.pod_num * self.spine_num_per_pod
                start_nodes.append(a_it_node)
                end_nodes.append(supply_node_id_k0)
                capacities.append(min(logical_spine_oxc_link[i, t, 0], logical_a_it[i, t]))
                unit_costs.append(-1)
                if logical_spine_oxc_link[i, t, 0] + logical_spine_oxc_link[i, t, 1] != logical_a_it[i, t]:
                    print("debug logical_spine_oxc_link", logical_spine_oxc_link[i, t, 0],
                          logical_spine_oxc_link[i, t, 1], logical_a_it[i, t])
                assert logical_spine_oxc_link[i, t, 0] + logical_spine_oxc_link[i, t, 1] == logical_a_it[i, t]
                supplies_map[f'{i}_{t}'] = min(logical_spine_oxc_link[i, t, 0], logical_a_it[i, t])

        for j in range(self.pod_num):
            for t in range(self.spine_num_per_pod):
                b_jt_node = self.spine_num_per_pod * j + t + 3 * self.pod_num * self.spine_num_per_pod
                end_node_id_k0 = self.spine_num_per_pod * j + t + 2 * self.pod_num * self.spine_num_per_pod
                start_nodes.append(end_node_id_k0)
                end_nodes.append(b_jt_node)
                capacities.append(min(logical_spine_oxc_link[j, t, 0], logical_b_jt[j, t]))
                unit_costs.append(-1)
                assert logical_spine_oxc_link[j, t, 0] + logical_spine_oxc_link[j, t, 1] == logical_b_jt[j, t]
                supplies_map[f'{3 * self.pod_num + j}_{t}'] = -min(logical_spine_oxc_link[j, t, 0], logical_b_jt[j, t])

        supplies = []
        for i in range(4 * self.pod_num):
            for t in range(self.spine_num_per_pod):
                if f'{i}_{t}' in supplies_map:
                    supplies.append(supplies_map[f'{i}_{t}'])  # 求x_ij0t
                else:
                    supplies.append(0)

        # Instantiate a SimpleMinCostFlow solver.
        min_cost_flow_ = min_cost_flow.SimpleMinCostFlow()
        # Add each arc.
        for i in range(0, len(start_nodes)):
            min_cost_flow_.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i],
                                                                capacities[i], unit_costs[i])

        # print("supply:")
        # Add node supplies.
        for i in range(0, len(supplies)):
            min_cost_flow_.set_node_supply(i, supplies[i])

        # Find the minimum cost flow between node 0 and node 4.
        if min_cost_flow_.solve() == min_cost_flow_.OPTIMAL:
            x_ijkt = np.zeros((self.pod_num, self.pod_num, 2, self.spine_num_per_pod), dtype=int)
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for t in range(self.spine_num_per_pod):
                        for local_arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}_{t}'])):
                            if arc_capacity_k0_map[f'{i}_{j}_{t}'][local_arc_id] > 0:
                                global_arc_id = i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{t}_{local_arc_id}_0']
                                x_ijkt[i, j, 0, t] += min_cost_flow_.flow(global_arc_id)

            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for t in range(self.spine_num_per_pod):
                        assert x_ijkt[i, j, 0, t] <= c_ijt[i, j, t]
                        x_ijkt[i, j, 1, t] = c_ijt[i, j, t] - x_ijkt[i, j, 0, t]
                        assert x_ijkt[i, j, 0, t] >= 0

            return x_ijkt
        else:
            success_MCF = False
            assert success_MCF
            print('There was an issue with the min cost flow input.')
