import time
import logging
from collections import defaultdict
import numpy as np
import os
import warnings
from copy import deepcopy
import random
from rapidAIsim.scheduler.ocsexpander.cijt_solver import CijtSolver
from rapidAIsim.scheduler.ocsexpander.mcf_solver import MCFSolver
from rapidAIsim.scheduler.ocsexpander.greedy_solver import GreedySolver
from rapidAIsim.scheduler.ocsexpander.routing_solver import RoutingSolver
from rapidAIsim.scheduler.ocsexpander import mesh_solver, TE_solver_lp, TE_solver, gpu_placement, divide_oxc_matrix,divide_oxc_matrix_mcf
from rapidAIsim.communication_strategy.all2all import All2All
from rapidAIsim.communication_strategy.ring import Ring
from rapidAIsim.core.event.RepairEvent import RepairEvent
from rapidAIsim.core.network_refresh import handle_task_finish_immediately

logging.basicConfig(level=logging.ERROR)


class OCSExpander:
    def __init__(self, ocs_reconfiguration=True):
        # static variable
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        self.ocs_num = infra.ocs_num
        self.pod_num = infra.pod_num
        self.spine_num_per_pod = infra.spine_num_per_pod
        self.spine_up_port_num = infra.spine_up_port_num
        self.gpu_per_server = infra.NIC_num_in_a_server
        self.server_num_per_pod = infra.server_num_per_pod
        self.server_per_leaf = infra.server_per_leaf
        self.gpu_per_pod = self.gpu_per_server * self.server_num_per_pod
        self.gpu_per_leaf = self.gpu_per_server * self.server_per_leaf
        self.TP = self.gpu_per_server
        self.leaf_num_per_pod = infra.leaf_num_per_pod
        self.spine_oxc_link_num = 1
        self.total_leaf_num = infra.leaf_switch_num
        # dynamical variable
        self.T_a_b = np.zeros((self.pod_num, self.pod_num), dtype=int)
        # self.cur_real_link_demand = np.zeros((self.pod_num, self.pod_num), dtype=int)
        self.u_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_num_per_pod), dtype=int)
        self.translate_link_new(self.u_ijkt)
        self.job_flow_demand_map = {}
        self.allocated_link_mapping = None
        self.ocs_reconfiguration = ocs_reconfiguration

        self.global_leaf_comm = np.zeros((self.total_leaf_num, self.total_leaf_num), dtype=int)
        self.scaled_global_leaf_comm = deepcopy(self.global_leaf_comm)
        # self.routing_solver = RoutingSolver(self.pod_num, self.spine_num_per_pod)
        self.gpu_placement_scheduler = gpu_placement.GPUPlacement()

        self.checkpoint_time = 60
        self.job_start_time_map = {}
        self.failure_id_gpu_status_map = {}
        
        self.ilp_false_map = {}




    def schedule(self, TP, DP, PP, EP, task_id, model_size):
        from rapidAIsim.core.simulator import Simulator
        Simulator.task_has_computation_time[task_id] = 0.
        Simulator.task_has_communication_size[task_id] = 0.
        Simulator.task_need_comp_time[task_id] = 0.
        Simulator.task_need_comm_size[task_id] = 0.
        Simulator.task_expected_comm_time[task_id] = 0.
        Simulator.task_actual_comm_time[task_id] = 0.
        # base_time = time.time()
        print(f'task {task_id} with gpu size {TP * DP * PP} and EP is {EP}')

        # Step 1 gpu placement in each pod
        require_server_num = TP * DP * PP // self.gpu_per_server
        pod_used_server_list, job_gpu_used_list = self.gpu_placement_scheduler.occupy_resource(require_server_num, EP,
                                                                                               task_id)
        if len(pod_used_server_list) == 0:
            return False, None, None, None, None
       
        # Step 3 intra leaf flow demand generator
        if Simulator.CONF_DICT['need_comm_orc'] == 'no':
            dp_pp_array, ep_array = self.generate_flow_demand_no_arrangement(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list)
        else:
            dp_pp_array, ep_array = self.generate_flow_demand(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list)
        ep_qp, pp_qp, dp_qp = self.generate_qp(ep_array, dp_pp_array)
        # print("stage 3 cost ", time.time() - base_time)
        if self.ocs_reconfiguration is False:
            return True, ep_qp, pp_qp, dp_qp, self.allocated_link_mapping
        
        if Simulator.CONF_DICT['strategy'] == 'dragonfly':
            print("debug dragonfly")
            x_ijkt = deepcopy(self.u_ijkt)
            self.allocated_link_mapping = self.translate_link_new(x_ijkt)
            return True, ep_qp, pp_qp, dp_qp, self.allocated_link_mapping
        
        if Simulator.CONF_DICT['rail_optimized'] == 'yes':
            self.job_flow_demand_map[task_id] = self.generate_leaf_communication_rail_optimized(pp_qp, dp_qp)
            self.global_leaf_comm += self.job_flow_demand_map[task_id]
            self.scaled_global_leaf_comm = self.scaling_leaf_comm()
            leaf_ij_copy = deepcopy(self.scaled_global_leaf_comm)
        else:
            self.job_flow_demand_map[task_id] = self.generate_leaf_communication(pp_qp, dp_qp)
            self.global_leaf_comm += self.job_flow_demand_map[task_id]
            leaf_ij_copy = deepcopy(self.global_leaf_comm)
            
        # Step 4.2 use c++ to calculate the intra spine flow demand
        leaf_spine_link_num = Simulator.get_infrastructure().leaf_spine_link_num
        if Simulator.CONF_DICT['strategy'] == 'no_balance':
            print("debug no_balance")
            base_time = time.time()
            c_ijt = self.generate_pod_level_link(leaf_ij_copy)
            print("time cost logical topo:", time.time() - base_time)
        else:
            leaf_ij_copy, leaf_ij_copy_T = divide_oxc_matrix.solve(leaf_ij_copy, self.total_leaf_num)
            solver = CijtSolver(leaf_ij_copy, self.spine_num_per_pod, self.pod_num * self.leaf_num_per_pod)
            base_time = time.time()
            l_abt = solver.solve()
            print("time cost logical topo:", time.time() - base_time)
            for t in range(self.spine_num_per_pod):
                l_abt[:, :, t] += l_abt[:, :, t].T
            # Step 4.3 do cross-link and TE-solver
            c_ijt, c_jit, flag, init_c_ijt = self.generate_link_demand(l_abt,task_id)  # 生成leaf demand
            if flag == False:
                return False, None, None, None, None

        print("debug c_ijt",c_ijt.shape)
        base_time = time.time()
   
        if Simulator.CONF_DICT['strategy'] == 'no_balance':
            new_cijt = c_ijt 
        else:
            new_cijt = c_ijt + c_jit
        if Simulator.CONF_DICT['strategy'] == 'bvn':
            x_ijkt = self.generate_ocs_configuration_bvn(new_cijt)
        elif Simulator.CONF_DICT['strategy'] == 'ilp':
            x_ijkt = self.generate_ocs_configuration_ilp(new_cijt)
        elif Simulator.CONF_DICT['strategy'] == 'ilp_itv':
            x_ijkt = self.generate_ocs_configuration_ilp_itv(c_ijt)
        elif Simulator.CONF_DICT['strategy'] == 'helios2':
            x_ijkt = self.generate_ocs_configuration_helios(new_cijt)
        else:
            x_ijkt = self.generate_ocs_configuration(c_ijt)

        #TODO SCHEDULER_TIME_COST
        Simulator.SCHEDULER_TIME_COST[task_id] = time.time() - base_time
        gpu_num = Simulator.get_infrastructure().NIC_num


        print("solve time cost:", Simulator.SCHEDULER_TIME_COST[task_id])
        
        self.u_ijkt = deepcopy(x_ijkt)
                        
        if Simulator.CONF_DICT['strategy'] == 'bvn' or Simulator.CONF_DICT['strategy'] == 'ilp' or Simulator.CONF_DICT['strategy'] == 'helios2' or (task_id in self.ilp_false_map and self.ilp_false_map[task_id] == True):
            print("debug greedy ocs")
            allocated_link_mapping = self.translate_link_bvn(self.u_ijkt, new_cijt, init_c_ijt)
        else:
            allocated_link_mapping = self.translate_link_new(self.u_ijkt, new_cijt)      


        total_link = 0
        realize_link = 0
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    total_link += int(new_cijt[i,j,t])
                    if np.sum(self.u_ijkt[i,j,:,t]) < new_cijt[i,j,t]:
                        realize_link += int(np.sum(self.u_ijkt[i,j,:,t]))
                    else:
                        realize_link += int(new_cijt[i,j,t])

        
        
        
        self.allocated_link_mapping = allocated_link_mapping
        self.job_start_time_map[task_id] = Simulator.get_current_time()
        return True, ep_qp, pp_qp, dp_qp, allocated_link_mapping

    def generate_pod_level_link(self, leaf_ij):
        C_pod = np.zeros((self.pod_num, self.pod_num), dtype=int)
        from rapidAIsim.core.simulator import Simulator
        leaf_spine_link_num = int(Simulator.CONF_DICT['leaf_spine_link_num'])
        # Calculate the communication matrix at pod level
        T = self.leaf_num_per_pod
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                if i == j:
                    continue  
                for a in range(i * T, (i + 1) * T):
                    for b in range(j * T, (j + 1) * T):
                        C_pod[i][j] += leaf_ij[a][b]
        
        # print("debug C_pod", self.spine_up_port_num)    
        # with np.printoptions(threshold=np.inf):
        #     print(C_pod)

        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver('Gurobi')
        c_ijt = np.empty((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=pywraplp.Variable)
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    c_ijt[i, j, t] = solver.IntVar(0, self.spine_up_port_num, f'c_{i}_{j}_{t}')
        obj_ij = np.empty((self.pod_num, self.pod_num), dtype=pywraplp.Variable)
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                obj_ij[i, j] = solver.IntVar(0, self.spine_up_port_num*self.pod_num, f'obj_ij_{i}_{j}')
                
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                # solver.Add(solver.Sum(c_ijt[i, j, :].ravel().tolist()) == C_pod[i,j])
                solver.Add(obj_ij[i, j] >= solver.Sum(c_ijt[i, j, :].ravel().tolist()) - C_pod[i,j])
                solver.Add(obj_ij[i, j] >= C_pod[i,j] - solver.Sum(c_ijt[i, j, :].ravel().tolist()))
                
        for i in range(self.pod_num):
            for t in range(self.spine_num_per_pod):
                # solver.Add(solver.Sum(c_ijt[i, :, t].ravel().tolist()) <= self.spine_num_per_pod)
                solver.Add(solver.Sum(c_ijt[i, :, t].ravel().tolist()) <= self.spine_up_port_num) #leaf_spine_link_num
            
        for j in range(self.pod_num):
            for t in range(self.spine_num_per_pod):
                solver.Add(solver.Sum(c_ijt[:, j, t].ravel().tolist()) <= self.spine_up_port_num)
            
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    solver.Add(c_ijt[i, j, t] == c_ijt[j, i, t] )

        obj = solver.Sum(obj_ij.ravel().tolist())
        solver.Minimize(obj)

        
        # 模型求解
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            assert False
        # 获取结果
        def get_solution_variable(x):
            return x.solution_value()
        get_solution_variable = np.vectorize(get_solution_variable)
        init_C_ijt = get_solution_variable(c_ijt)
        print("start TE_solver")
        c_ijt = self.calculate_max_min_fairness(init_C_ijt)
        # with np.printoptions(threshold=np.inf):
        #     for i in range(self.pod_num):
        #         for t in range(self.spine_num_per_pod):
        #             print(np.sum(c_ijt[i, :, t]))
        #     for i in range(self.pod_num):
        #         for t in range(self.spine_num_per_pod):
        #             print(np.sum(c_ijt[:, i, t]))
        tmp_C_ijt = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        tmp_C_ijt_T = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        for t in range(self.spine_num_per_pod):
            tmp_res, tmp_res_T = divide_oxc_matrix.solve(c_ijt[:, :, t], self.pod_num)
            init_tmp_res, init_tmp_res_T = divide_oxc_matrix.solve(init_C_ijt[:, :, t], self.pod_num)
            tmp_res = TE_solver.solve(self.pod_num, self.spine_up_port_num // 2, tmp_res, init_tmp_res)
            tmp_C_ijt[:, :, t] += tmp_res
            tmp_C_ijt_T[:, :, t] += tmp_res.T
        c_ijt_solution = tmp_C_ijt 

        return c_ijt_solution
        

    @staticmethod
    def generate_qp(ep_array, dp_pp_array):
        """
        根据输入生成EP、PP、DP的QP。
        :param ep_array: EP的GPU分配
        :param dp_pp_array: DP和PP的GPU分配
        :return: EP、PP、DP的QP。其中EP是个二维数组，第一个维度是round，代表每个stage的所有EP域的通信对。
        PP是一个二维数组，表示一轮训练的所有前向和反向过程（按前向、反向的顺序）。第一个维度是round，代表当前阶段的所有PP域的通信对。
        DP是一个二维数组，表示一轮训练的所有DP通信对。第一个维度是从上到下的每个DP域（前向传播中的PP顺序），
        第二个维度代表DP通信的一个round。因为每个DP round都是相同的，所以这里省略。
        """
        # 生成EP的QP
        PP, DP, TP = dp_pp_array.shape
        ep_nums, ep_size = ep_array.shape[0], ep_array.shape[1]
        if ep_size == TP:
            ep_qp = []
        else:
            ep_qp = [[] for _ in range(ep_size // TP - 1)]
            for i in range(ep_nums):
                gpu_list = ep_array[i, :]
                # 分平面通信
                for j in range(TP):
                    gpu_list_face = gpu_list[j:][::TP]
                    # 这里communication_size设置成0原因是为了复用All2All的代码，只是为了生成QP，不需要真实的communication_size
                    ep_round_pairs = All2All.get_pairwise_every_round_pair(len(gpu_list_face), 0)
                    for round_id in range(len(ep_round_pairs)):
                        for pair in ep_round_pairs[round_id]:
                            ep_qp[round_id].append((gpu_list_face[pair[0]], gpu_list_face[pair[1]]))

        # 生成PP的QP。按照先前向，后反向的顺序生成QP，所以一共包含2*(PP-1)个round
        
        pp_qp = []
        for i in range(PP - 1):
            pp = []
            for j in range(DP):
                for k in range(TP):
                    pp.append((dp_pp_array[i, j, k], dp_pp_array[i + 1, j, k]))
            pp_qp.append(pp)
        for i in range(PP - 1, 0, -1):
            pp = []
            for j in range(DP):
                for k in range(TP):
                    pp.append((dp_pp_array[i, j, k], dp_pp_array[i - 1, j, k]))
            pp_qp.append(pp)

        # 生成DP的QP
        if DP == 1:
            return ep_qp, pp_qp, []
        dp_qp_pairs = Ring.get_ring_every_round_pair(DP, 0)[0]
        qp_num = len(dp_qp_pairs)
        reverse_dp_round_pairs = dp_qp_pairs[qp_num // 2:] + dp_qp_pairs[:qp_num // 2]
        from rapidAIsim.core.simulator import Simulator
        dp_qp = [[] for _ in range(PP)]
        if Simulator.CONF_DICT['rail_optimized'] == 'yes':
            for i in range(PP):
                for j in range(TP // 2):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in dp_qp_pairs[:qp_num // 2]])
                for j in range(TP // 2):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in reverse_dp_round_pairs[:qp_num // 2]])
        else:
            for i in range(PP):
                for j in range(TP // 2):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in dp_qp_pairs[:qp_num // 2]])
                for j in range(TP // 2, TP):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in reverse_dp_round_pairs[:qp_num // 2]])
        return ep_qp, pp_qp, dp_qp
    
    @staticmethod
    def generate_flow_demand_no_arrangement(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list):
        # remain_job_gpu_used_list = deepcopy(job_gpu_used_list)
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        pod_num = infra.pod_num
        server_num_per_pod = infra.server_num_per_pod
        server_per_leaf = infra.server_per_leaf
        gpu_per_server = infra.NIC_num_in_a_server
        pod_server_pair = []
        for pod_id in range(pod_num):
            if pod_used_server_list[pod_id] > 0:
                pod_server_pair.append((pod_id, pod_used_server_list[pod_id]))
        pod_server_pair.sort(key=lambda x: x[1])
        PP_DP_domain_matrix = np.zeros((PP, DP), dtype=int)
        down_flag = True

        curr_DP = 0
        curr_PP = 0
        right_flag = True  # 新增标志，用于控制列的移动方向
        for pod_id, server_num in pod_server_pair:
            for _ in range(server_num):
                PP_DP_domain_matrix[curr_PP, curr_DP] = pod_id
                if right_flag:
                    curr_DP += 1
                else:
                    curr_DP -= 1
                if curr_DP == DP:  # 假设 DP 是矩阵的列数
                    curr_DP = DP - 1
                    curr_PP += 1
                    right_flag = False
                elif curr_DP == -1:
                    curr_DP = 0
                    curr_PP += 1
                    right_flag = True

        gpu_num_per_leaf = gpu_per_server * server_per_leaf
        gpu_num_per_pod = gpu_per_server * server_num_per_pod
        # 构建一个从TP到server_list的映射
        # job_gpu_used_list标注了该任务在每个Pod，每个leaf中用到的GPU
        dp_pp_array = np.zeros((PP, DP, TP), dtype=int)
        select_gpus = np.where(job_gpu_used_list == 1)
        tmp_pod_id, tmp_leaf_id, k = select_gpus
        gpu_indices = np.array(select_gpus, dtype=int).T
        num_selected_gpus = len(gpu_indices)
        global_gpu_ids = np.zeros(num_selected_gpus, dtype=int)
        global_gpu_ids += gpu_indices[:, 0] * gpu_num_per_pod + gpu_indices[:, 1] * gpu_num_per_leaf + gpu_indices[:, 2]

        select_pods = [pair[0] for pair in pod_server_pair]
        gpu_indices = {pod: gpu_indices[tmp_pod_id == pod, :].tolist() for pod in select_pods}
        for pod in select_pods:
            gpu_indices[pod].reverse()

        for (curr_PP, curr_DP), chosen_pod_id in np.ndenumerate(PP_DP_domain_matrix):
            select_gpu_indices = []
            for _ in range(TP):
                index = gpu_indices[chosen_pod_id].pop()
                select_gpu_indices.append(index)
            select_gpu_indices = np.array(select_gpu_indices, dtype=int)
            select_gpu_ids = np.zeros(TP, dtype=int)
            select_gpu_ids += select_gpu_indices[:, 0] * gpu_num_per_pod + \
                              select_gpu_indices[:, 1] * gpu_num_per_leaf + select_gpu_indices[:, 2]
            dp_pp_array[curr_PP, curr_DP, :] = select_gpu_ids
        assert not len(np.where(dp_pp_array == 0)[0]) >= TP

        # 生成EP流量
        EP_domain_num = num_selected_gpus // EP
        ep_array = np.zeros((EP_domain_num, EP), dtype=int)
        if EP <= TP:
            return dp_pp_array, ep_array

        EP_ids = np.arange(num_selected_gpus) // EP
        gpu_indices_within_ep = np.arange(num_selected_gpus) % EP
        ep_array[EP_ids, gpu_indices_within_ep] = global_gpu_ids
        ep_array_pod = ep_array // gpu_num_per_pod
        if not np.all(ep_array_pod == ep_array_pod[:, [0]]):
            warning_text = "EP communication is not in the same pod. The task info is:\n"
            warning_text += f"TP: {TP}, DP: {DP}, PP: {PP}, EP: {EP}\n"
            warning_text += f"pod_server_pair: {pod_server_pair}\n"
            warning_text += f"PP_DP_domain_matrix: {PP_DP_domain_matrix}\n"
            warning_text += f"job_gpu_used_list: {job_gpu_used_list}\n"
            warning_text += f"dp_pp_array: {dp_pp_array}\n"
            warning_text += f"ep_array: {ep_array}\n"
            warning_text += f"ep_array_pod: {ep_array_pod}\n"
            warnings.warn("EP communication is not in the same pod.")
            print(warning_text)
        return dp_pp_array, ep_array

    @staticmethod
    def generate_flow_demand(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list):
        # remain_job_gpu_used_list = deepcopy(job_gpu_used_list)
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        pod_num = infra.pod_num
        server_num_per_pod = infra.server_num_per_pod
        server_per_leaf = infra.server_per_leaf
        gpu_per_server = infra.NIC_num_in_a_server
        pod_server_pair = []
        for pod_id in range(pod_num):
            if pod_used_server_list[pod_id] > 0:
                pod_server_pair.append((pod_id, pod_used_server_list[pod_id]))
        pod_server_pair.sort(key=lambda x: x[1])
        PP_DP_domain_matrix = np.zeros((PP, DP), dtype=int)
        down_flag = True

        curr_DP = 0
        curr_PP = 0
        for pod_id, server_num in pod_server_pair:
            for _ in range(server_num):
                PP_DP_domain_matrix[curr_PP, curr_DP] = pod_id
                if down_flag:
                    curr_PP += 1
                else:
                    curr_PP -= 1
                if curr_PP == PP:
                    curr_PP = PP - 1
                    curr_DP += 1
                    down_flag = False
                elif curr_PP == -1:
                    curr_PP = 0
                    curr_DP += 1
                    down_flag = True

        gpu_num_per_leaf = gpu_per_server * server_per_leaf
        gpu_num_per_pod = gpu_per_server * server_num_per_pod
        # 构建一个从TP到server_list的映射
        # job_gpu_used_list标注了该任务在每个Pod，每个leaf中用到的GPU
        dp_pp_array = np.zeros((PP, DP, TP), dtype=int)
        select_gpus = np.where(job_gpu_used_list == 1)
        tmp_pod_id, tmp_leaf_id, k = select_gpus
        gpu_indices = np.array(select_gpus, dtype=int).T
        num_selected_gpus = len(gpu_indices)
        global_gpu_ids = np.zeros(num_selected_gpus, dtype=int)
        global_gpu_ids += gpu_indices[:, 0] * gpu_num_per_pod + gpu_indices[:, 1] * gpu_num_per_leaf + gpu_indices[:, 2]

        select_pods = [pair[0] for pair in pod_server_pair]
        gpu_indices = {pod: gpu_indices[tmp_pod_id == pod, :].tolist() for pod in select_pods}
        for pod in select_pods:
            gpu_indices[pod].reverse()

        for (curr_PP, curr_DP), chosen_pod_id in np.ndenumerate(PP_DP_domain_matrix):
            select_gpu_indices = []
            for _ in range(TP):
                index = gpu_indices[chosen_pod_id].pop()
                select_gpu_indices.append(index)
            select_gpu_indices = np.array(select_gpu_indices, dtype=int)
            select_gpu_ids = np.zeros(TP, dtype=int)
            select_gpu_ids += select_gpu_indices[:, 0] * gpu_num_per_pod + \
                              select_gpu_indices[:, 1] * gpu_num_per_leaf + select_gpu_indices[:, 2]
            dp_pp_array[curr_PP, curr_DP, :] = select_gpu_ids
        assert not len(np.where(dp_pp_array == 0)[0]) >= TP

        # 生成EP流量
        EP_domain_num = num_selected_gpus // EP
        ep_array = np.zeros((EP_domain_num, EP), dtype=int)
        if EP <= TP:
            return dp_pp_array, ep_array

        EP_ids = np.arange(num_selected_gpus) // EP
        gpu_indices_within_ep = np.arange(num_selected_gpus) % EP
        ep_array[EP_ids, gpu_indices_within_ep] = global_gpu_ids
        ep_array_pod = ep_array // gpu_num_per_pod
        if not np.all(ep_array_pod == ep_array_pod[:, [0]]):
            warning_text = "EP communication is not in the same pod. The task info is:\n"
            warning_text += f"TP: {TP}, DP: {DP}, PP: {PP}, EP: {EP}\n"
            warning_text += f"pod_server_pair: {pod_server_pair}\n"
            warning_text += f"PP_DP_domain_matrix: {PP_DP_domain_matrix}\n"
            warning_text += f"job_gpu_used_list: {job_gpu_used_list}\n"
            warning_text += f"dp_pp_array: {dp_pp_array}\n"
            warning_text += f"ep_array: {ep_array}\n"
            warning_text += f"ep_array_pod: {ep_array_pod}\n"
            warnings.warn("EP communication is not in the same pod.")
            print(warning_text)
        return dp_pp_array, ep_array

    def generate_link_demand(self, l_abt, taskid=-1):
        from rapidAIsim.core.simulator import Simulator
        print("start calculate init spine communication")
        init_C_ijt = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        for start_leaf in range(self.total_leaf_num):
            for end_leaf in range(self.total_leaf_num):
                if np.sum(l_abt[start_leaf, end_leaf, :]) == 0:
                    continue
                start_pod = start_leaf // self.leaf_num_per_pod
                end_pod = end_leaf // self.leaf_num_per_pod
                if start_pod == end_pod:
                    continue
                init_C_ijt[start_pod, end_pod, :] += l_abt[start_leaf, end_leaf, :]

        print("start TE_solver")
        c_ijt = self.calculate_max_min_fairness(init_C_ijt)
        print("start cross link")
        tmp_C_ijt = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        tmp_C_ijt_T = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        for t in range(self.spine_num_per_pod):
            tmp_res, tmp_res_T = divide_oxc_matrix.solve(c_ijt[:, :, t], self.pod_num)
            init_tmp_res, init_tmp_res_T = divide_oxc_matrix.solve(init_C_ijt[:, :, t], self.pod_num)
            if Simulator.CONF_DICT['need_comm_orc'] == 'no' or (int(Simulator.CONF_DICT['leaf_spine_link_num']) == 1 and self.server_per_leaf == 1):
                tmp_res,flag = TE_solver.solve(self.pod_num, self.spine_up_port_num // 2, tmp_res, init_tmp_res, True)
            else:
                tmp_res,flag = TE_solver.solve(self.pod_num, self.spine_up_port_num // 2, tmp_res, init_tmp_res, False)
            if flag == False:
                return None, False
            tmp_C_ijt[:, :, t] += tmp_res
            tmp_C_ijt_T[:, :, t] += tmp_res.T
        # for i in range(self.pod_num):
        #     for j in range(self.pod_num):
        #         for t in range(self.spine_num_per_pod):
        #             if tmp_C_ijt[i,j,t]<init_C_ijt[i,j,t]:
        #                 print("debug no require link",tmp_C_ijt[i,j,t],init_C_ijt[i,j,t],self.spine_num_per_pod*i+t,self.spine_num_per_pod*j+t)
        
        # print("debug logical topology for task",taskid)
        # with np.printoptions(threshold=np.inf):
        #     print(np.sum(tmp_C_ijt,axis=-1))
        # with np.printoptions(threshold=np.inf):
        #     print(np.sum(init_C_ijt,axis=-1))
        return tmp_C_ijt, tmp_C_ijt_T, True, init_C_ijt

    def generate_ocs_configuration(self, c_ijt):
        start = time.time()
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['basic_mcf'] == "yes":
            print("debug basic_mcf")
            oxc_list = list(range(self.ocs_num))
            m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                                c_ijt, self.u_ijkt, self.spine_up_port_num)
            x_ijkt = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size * 2, m_solver.spine_num_per_pod),
                          dtype=float)
            x_ijkt_star = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size, m_solver.spine_num_per_pod),
                            dtype=float)
            for t in range(self.spine_num_per_pod):
                p_ij = c_ijt[:,:,t]
                solver = CijtSolver(p_ij, m_solver.oxc_physical_group_size, self.pod_num)
                p_ijk = solver.solve()
                # print("debug self.ocs_num",p_ijk.shape,self.ocs_num,self.pod_num)
                x_ijkt_star[:, :, :, t] = p_ijk
                
            for k in range(0, m_solver.oxc_physical_group_size, 1):
                x_ijkt[:, :, 2 * k, :] += x_ijkt_star[:, :, k, :]
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for k in range(0, self.spine_up_port_num, 2):
                        x_ijkt[j, i, k + 1, :] = x_ijkt[i, j, k, :]
        else:
            # print("start mcf")
            oxc_list = list(range(self.ocs_num))
            m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                                c_ijt, self.u_ijkt, self.spine_up_port_num)
            x_ijkt = m_solver.solve()
        end = time.time()

        print("mcf time cost:", end - start)
        return x_ijkt
    


    def generate_ocs_configuration_helios(self, c_ijt):
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                             c_ijt, self.u_ijkt, self.spine_up_port_num)
        x_ijkt = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size * 2, m_solver.spine_num_per_pod),
                          dtype=float)
        from rapidAIsim.scheduler.ocsexpander.helios import bg

        for t in range(m_solver.spine_num_per_pod):
            d_wave = c_ijt[:, :, t]
            base_capacity = 1
            r = self.ocs_num
            x_ijk = bg(r, d_wave, base_capacity)
            x_ijkt[:, :, :, t] += x_ijk

        return x_ijkt

    def generate_ocs_configuration_helios2(self, c_ijt):
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                             c_ijt, self.u_ijkt, self.spine_up_port_num)
        x_ijkt = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size * 2, m_solver.spine_num_per_pod),
                          dtype=float)
        x_ijkt_star = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size, m_solver.spine_num_per_pod),
                          dtype=float)
        from rapidAIsim.scheduler.ocsexpander.helios import bg

        for t in range(m_solver.spine_num_per_pod):
            d_wave = c_ijt[:, :, t]
            base_capacity = 1
            r = self.ocs_num
            x_ijk = bg(r, d_wave, base_capacity)
            x_ijkt_star[:, :, :, t] += x_ijk

        for k in range(0, m_solver.oxc_physical_group_size, 1):
            x_ijkt[:, :, 2 * k, :] += x_ijkt_star[:, :, k, :]

        # self.check_tmp_MCF(self.A_ij, original_oxc_to_each_spine_num)
        for i in range(self.pod_num):
            for j in range(self.pod_num):
                for k in range(0, self.spine_up_port_num, 2):
                    x_ijkt[j, i, k + 1, :] = x_ijkt[i, j, k, :]

        return x_ijkt

        

    def generate_ocs_configuration_ilp(self, c_ijt):
        flag, x_ijkt = mesh_solver.solve(self.spine_num_per_pod, self.spine_up_port_num, c_ijt, self.u_ijkt, False)
        if flag == False:
            print("ILP False")
            return self.generate_ocs_configuration_bvn(c_ijt)
        else:
            print("ILP True")
        return x_ijkt
    
    def generate_ocs_configuration_ilp_itv(self, c_ijt):
        flag, x_star_ijkt = mesh_solver.solve(self.spine_num_per_pod, self.spine_up_port_num//2, c_ijt, self.u_ijkt, True)
        
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                             c_ijt, self.u_ijkt, self.spine_up_port_num)
        x_ijkt = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size * 2, m_solver.spine_num_per_pod),
                          dtype=float)
            
        if flag == False:
            print("ILP False")
            x_ijkt = self.generate_ocs_configuration_bvn_itv(c_ijt)
        else:
            print("ILP True")
            for k in range(0, m_solver.oxc_physical_group_size, 1):
                x_ijkt[:, :, 2 * k, :] += x_star_ijkt[:, :, k, :]

            # self.check_tmp_MCF(self.A_ij, original_oxc_to_each_spine_num)
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for k in range(0, self.spine_up_port_num, 2):
                        x_ijkt[j, i, k + 1, :] = x_ijkt[i, j, k, :]
        return x_ijkt
    

    def normalize_matrix(self, matrix):
        # 检查输入是否为二维矩阵
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        # 检查行和与列和是否相等
        if not np.allclose(row_sums, col_sums):
            raise ValueError("Row sums and column sums are not equal.")
        total_sum = row_sums[0]  # 或者 col_sums[0]，因为它们应该相等
        # 归一化矩阵
        normalized_matrix = matrix / total_sum
        return normalized_matrix, total_sum


    def generate_res_array(self, W, S, r):
        import tensorflow as tf
        t, k, _ = S.shape

        # 归一化权重
        W_normalized = W / tf.reduce_sum(W)
        
        num_occurrences = tf.cast(tf.floor(W_normalized * r), dtype=tf.int32)
        
        total_occurrences = tf.reduce_sum(num_occurrences)
        remaining = r - total_occurrences
        
        sorted_indices = tf.argsort(W_normalized, direction='DESCENDING')
        for i in range(remaining):
            num_occurrences = tf.tensor_scatter_nd_add(
                num_occurrences,
                indices=[[sorted_indices[i]]],
                updates=[1]
            )
        
        res_array_list = []
        for i in range(t):
            res_array_list.extend([S[i]] * num_occurrences[i].numpy())
        
        res_array = tf.stack(res_array_list)
        
        # 转换为NumPy数组
        res_array_np = res_array.numpy().astype(np.int64)

        res_array_np = np.moveaxis(res_array_np, 0, -1)

        # for l in range(r):
        #     new_array = res_array_np[:,:,l]
        #     print(new_array)
        #     # print("debug_new_array")
        #     # print(new_array.shape)
        #     # 检查每一行的和
        #     for i, row in enumerate(new_array):
        #         assert sum(row) == 1, f"第{i+1}行的和不是1"
            
        #     # 检查每一列的和
        #     for j in range(len(new_array[0])):
        #         assert sum(new_array[i][j] for i in range(len(new_array[0]))) == 1, f"第{j+1}列的和不是1"
        
        return res_array_np


    def generate_ocs_configuration_bvn(self, c_ijt):

        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                             c_ijt, self.u_ijkt, self.spine_up_port_num)
        x_ijkt = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size * 2, m_solver.spine_num_per_pod),
                          dtype=int)
        import tensorflow as tf
        import bvn
        flag = False
        for t in range(m_solver.spine_num_per_pod):
            new_array = c_ijt[:, :, t]
            new_array_tf = tf.expand_dims(tf.constant(new_array, dtype=tf.float32, name='doubly_stochastic'),axis=0)
            p, c = bvn.bvn(new_array_tf, m_solver.oxc_physical_group_size * 2)
            matrix = p[0]
            coff = c[0]
            res_array_np = self.generate_res_array(coff, matrix, self.spine_up_port_num)
            x_ijkt[:, :, :, t] += res_array_np
        if flag:
            print("l2_incomp")

        return x_ijkt
    
    def generate_ocs_configuration_bvn_itv(self, c_ijt):
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                             c_ijt, self.u_ijkt, self.spine_up_port_num)
        x_ijkt = np.zeros((m_solver.pod_num, m_solver.pod_num, m_solver.oxc_physical_group_size * 2, m_solver.spine_num_per_pod),
                          dtype=int)
        import tensorflow as tf
        import bvn
        for t in range(m_solver.spine_num_per_pod):
            new_array = c_ijt[:, :, t]
            new_array_tf = tf.expand_dims(tf.constant(new_array, dtype=tf.float32, name='doubly_stochastic'),axis=0)
            p, c = bvn.bvn(new_array_tf, self.spine_up_port_num)
            matrix = p[0]
            coff = c[0]
            res_array_np = self.generate_res_array(coff, matrix, self.spine_up_port_num//2)
            for k in range(0, m_solver.oxc_physical_group_size, 1):
                x_ijkt[:, :, 2 * k, t] += res_array_np[:, :, k]

                # self.check_tmp_MCF(self.A_ij, original_oxc_to_each_spine_num)
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for k in range(0, m_solver.oxc_physical_group_size * 2, 2):
                        x_ijkt[j, i, k + 1, :] = x_ijkt[i, j, k, :]
        return x_ijkt

    def translate_link_new(self, x_ijkt, c_ijt=[]):
        from rapidAIsim.core.simulator import Simulator
        spine_num = self.spine_num_per_pod * self.pod_num
        nic_num = self.gpu_per_server * self.server_num_per_pod * self.pod_num
        try:
            leaf_spine_link_num = int(Simulator.CONF_DICT['leaf_spine_link_num'])
        except (KeyError, ValueError):
            leaf_spine_link_num = 1
        try:
            is_rail_optimized = Simulator.CONF_DICT['rail_optimized'] == 'yes'
        except KeyError:
            is_rail_optimized = False
        routing_solver = RoutingSolver(self.pod_num, spine_num, nic_num, self.server_num_per_pod,
                                       self.spine_up_port_num, leaf_spine_link_num, 1.0, is_rail_optimized)
        routing_solver.generate_routing_table(x_ijkt)
        Simulator.intra_pod_up_table = routing_solver.get_intra_pod_up_table()
        Simulator.intra_pod_down_table = routing_solver.get_intra_pod_down_table()
        Simulator.inter_pod_table = routing_solver.get_inter_pod_routing_table()
        Simulator.inter_pod_weighted_direct_table = routing_solver.get_inter_pod_weighted_direct_routing_table()
        Simulator.inter_pod_weighted_twohop_table = routing_solver.get_inter_pod_weighted_twohop_routing_table()
        allocated_link_mapping = routing_solver.get_connection_info_list()
        return allocated_link_mapping
    

    def translate_link_bvn(self, x_ijkt, c_ijt, init_c_ijt):
        from rapidAIsim.core.simulator import Simulator
        c_ij = np.sum(c_ijt, axis=2)
        flattened_indices = np.arange(c_ij.size)
        sorted_indices = flattened_indices[np.argsort(c_ij.ravel())]
        sorted_2d_indices = np.unravel_index(sorted_indices, c_ij.shape)
        
        g_ijkt = np.zeros_like(x_ijkt)

        # 记录需要处理的 (i, j, k, t) 索引对
        index_pairs = []
        
        for i, j in zip(*sorted_2d_indices):
                if i != j:
                    for k in range(x_ijkt.shape[2]):
                        for t in range(x_ijkt.shape[3]):
                            if x_ijkt[i, j, k, t] != x_ijkt[j, i, k, t]:
                                g_ijkt[i, j, k, t] = x_ijkt[i, j, k, t] - min(x_ijkt[i, j, k, t], x_ijkt[j, i, k, t])
                                x_ijkt[i, j, k, t] = min(x_ijkt[i, j, k, t], x_ijkt[j, i, k, t])
                                index_pairs.append((i, j, k, t))

        # for i, j, k, t in reversed(index_pairs):
        #     sum_over_j = np.sum(x_ijkt[i, :, k, t])
        #     sum_over_i = np.sum(x_ijkt[:, j, k, t])
        #     if sum_over_j == 0 and sum_over_i == 0:
        #         x_ijkt[i, j, k, t] += g_ijkt[i, j, k, t]
        #         x_ijkt[j, i, k, t] += g_ijkt[i, j, k, t]
                
        # 获取数组的形状
        a_size, b_size, k_size, t_size = x_ijkt.shape
        from collections import defaultdict
        Simulator.rerouting_link = defaultdict(int)
        leaf_num = self.leaf_num_per_pod * self.pod_num
        spine_num = self.spine_num_per_pod * self.pod_num
        nic_num = self.gpu_per_server * self.server_num_per_pod * self.pod_num
        # default 3-layer
        for a in range(a_size):
            for b in range(b_size):
                if np.sum(init_c_ijt[a, b, :]) > 0:
                    if np.sum(x_ijkt[b, a, :, :]) == 0:
                        x_ijkt[b, a, 0, 0] = 1
                        a_spine_id = nic_num + leaf_num + a*self.spine_num_per_pod
                        b_spine_id = nic_num + leaf_num + b*self.spine_num_per_pod
                        Simulator.rerouting_link[(b_spine_id, a_spine_id)] += 1
                    if np.sum(x_ijkt[a, b, :, :]) == 0:
                        x_ijkt[a, b, 0, 0] = 1
                        a_spine_id = nic_num + leaf_num + a*self.spine_num_per_pod
                        b_spine_id = nic_num + leaf_num + b*self.spine_num_per_pod
                        Simulator.rerouting_link[(a_spine_id, b_spine_id)] += 1
        # print("debug Simulator.rerouting_link")
        # print(Simulator.rerouting_link) 
        try:
            leaf_spine_link_num = int(Simulator.CONF_DICT['leaf_spine_link_num'])
        except (KeyError, ValueError):
            leaf_spine_link_num = 1
        try:
            is_rail_optimized = Simulator.CONF_DICT['rail_optimized'] == 'yes'
        except KeyError:
            is_rail_optimized = False
        routing_solver = RoutingSolver(self.pod_num, spine_num, nic_num, self.server_num_per_pod,
                                       self.spine_up_port_num, leaf_spine_link_num, 1.0, is_rail_optimized)
        routing_solver.generate_routing_table(x_ijkt)
        Simulator.intra_pod_up_table = routing_solver.get_intra_pod_up_table()
        Simulator.intra_pod_down_table = routing_solver.get_intra_pod_down_table()
        Simulator.inter_pod_table = routing_solver.get_inter_pod_routing_table()
        Simulator.inter_pod_weighted_direct_table = routing_solver.get_inter_pod_weighted_direct_routing_table()
        Simulator.inter_pod_weighted_twohop_table = routing_solver.get_inter_pod_weighted_twohop_routing_table()
        allocated_link_mapping = routing_solver.get_connection_info_list()
        return allocated_link_mapping
    
    
    

    def generate_leaf_communication(self, pp_qp, dp_qp):
        from rapidAIsim.core.simulator import Simulator
        pp_flow, dp_flow = set(), set()
        for pp in pp_qp:
            for pair in pp:
                src_leaf, dst_leaf = pair[0] // self.gpu_per_leaf, pair[1] // self.gpu_per_leaf
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod:
                    pp_flow.add((src_leaf, dst_leaf))
        for dp in dp_qp:
            for pair in dp:
                src_leaf, dst_leaf = pair[0] // self.gpu_per_leaf, pair[1] // self.gpu_per_leaf
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod:
                    dp_flow.add((src_leaf, dst_leaf))

        leaf_comm = np.zeros((self.pod_num * self.leaf_num_per_pod, self.pod_num * self.leaf_num_per_pod), dtype=int)
        leaf_spine_link_num = Simulator.get_infrastructure().leaf_spine_link_num
        # if leaf_spine_link_num == 1 and self.server_per_leaf == 1:
        #     for start_leaf, end_leaf in pp_flow:
        #         leaf_comm[start_leaf, end_leaf] += self.TP // 4
        #     for start_leaf, end_leaf in dp_flow:
        #         leaf_comm[start_leaf, end_leaf] += self.TP // 4
        # else:
        for start_leaf, end_leaf in pp_flow:
            leaf_comm[start_leaf, end_leaf] += self.TP
        for start_leaf, end_leaf in dp_flow:
            # leaf_comm[start_leaf, end_leaf] += self.TP // 2
            leaf_comm[start_leaf, end_leaf] += self.TP
        return leaf_comm

    def generate_leaf_communication_rail_optimized(self, pp_qp, dp_qp):
        from rapidAIsim.core.simulator import Simulator
        gpu_num = Simulator.get_infrastructure().NIC_num
        leaf_comm = np.zeros((self.pod_num * self.leaf_num_per_pod, self.pod_num * self.leaf_num_per_pod), dtype=int)
        intra_pod_up_table = Simulator.intra_pod_up_table
        for pp in pp_qp:
            for pair in pp:
                src_leaf, dst_leaf = intra_pod_up_table[pair[0]][0], intra_pod_up_table[pair[1]][0]
                src_leaf -= gpu_num
                dst_leaf -= gpu_num
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod:
                    leaf_comm[src_leaf, dst_leaf] += 1
        prev_dp_pod_comm = set()
        for dp in dp_qp:
            curr_dp_pod_comm = set()
            for pair in dp:
                src_leaf, dst_leaf = intra_pod_up_table[pair[0]][0], intra_pod_up_table[pair[1]][0]
                src_leaf -= gpu_num
                dst_leaf -= gpu_num
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod and (src_pod, dst_pod) not in prev_dp_pod_comm:
                    leaf_comm[src_leaf, dst_leaf] += 1
                    leaf_comm[dst_leaf, src_leaf] += 1
                    curr_dp_pod_comm.add((src_pod, dst_pod))
            prev_dp_pod_comm = curr_dp_pod_comm
        return leaf_comm

    def scaling_leaf_comm(self):
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        leaf_comm_sum = np.sum(self.global_leaf_comm, axis=0)
        leaf_up_port_num = infra.leaf_switch_port_num // 2
        filtered_indices = np.where(leaf_comm_sum > leaf_up_port_num)[0]
        arg_sort = filtered_indices[np.argsort(leaf_comm_sum[filtered_indices])]
        leaf_comm = np.triu(self.global_leaf_comm)
        for i in arg_sort:
            comm_sum = np.sum(leaf_comm[i, :]) + np.sum(leaf_comm[:, i])
            for j in range(i):
                if leaf_comm[j, i] > 0:
                    leaf_comm[j, i] = max(1, np.floor(leaf_comm[j, i] / comm_sum * leaf_up_port_num))
            for j in range(i + 1, self.total_leaf_num):
                if leaf_comm[i, j] > 0:
                    leaf_comm[i, j] = max(1, np.floor(leaf_comm[i, j] / comm_sum * leaf_up_port_num))
            new_comm_sum = np.sum(leaf_comm[i, :]) + np.sum(leaf_comm[:, i])
            temp_comm = np.zeros((self.total_leaf_num,), dtype=int)
            temp_comm[:i] = leaf_comm[:i, i]
            temp_comm[i + 1:] = leaf_comm[i, i + 1:]
            temp_arg_sort = np.argsort(temp_comm)
            temp_arg_sort = temp_arg_sort[::-1]
            if new_comm_sum > leaf_up_port_num:
                j = 0
                while new_comm_sum > leaf_up_port_num:
                    if temp_comm[temp_arg_sort[j]] > 1:
                        temp_comm[temp_arg_sort[j]] -= 1
                        new_comm_sum -= 1
                    j += 1
                    if j == self.total_leaf_num - 1:
                        j = 0
            assert new_comm_sum <= leaf_up_port_num
            # if new_comm_sum < leaf_up_port_num:
            #     j = 0
            #     while new_comm_sum < leaf_up_port_num:
            #         if leaf_comm[j, i] > 0:
            #             leaf_comm[j, i] += 1
            #             new_comm_sum += 1
            #         j += 1
            #     while new_comm_sum < leaf_up_port_num:
            #         if leaf_comm[i, j] > 0:
            #             leaf_comm[i, j] += 1
            #             new_comm_sum += 1
            #         j += 1
        leaf_comm = leaf_comm + leaf_comm.T
        assert np.all(np.sum(leaf_comm, axis=0) <= leaf_up_port_num)
        assert np.all(leaf_comm >= 0)
        return leaf_comm

    def update_finished_job(self, taskid, sim_time, waiting_list):
        """
        任务结束时，释放GPU资源，更新网络状态。当前版本中注释的内容(step2 & step3)是任务结束后仍然进行OCS重配置。
        """
        # Step 1 update demand
        from rapidAIsim.core.simulator import Simulator
        if self.ocs_reconfiguration and Simulator.CONF_DICT['strategy'] != 'dragonfly':
            self.global_leaf_comm -= self.job_flow_demand_map[taskid]

        # Step 4 release gpu
        self.gpu_placement_scheduler.release_resource(taskid)
        # return allocation_link_mapping

    def calculate_max_min_fairness(self, init_c_ijt):
        # c_ijt = deepcopy(init_c_ijt)
        c_ijt_copy = deepcopy(init_c_ijt)

        upper_triangular = np.triu(np.ones_like(c_ijt_copy, dtype=bool).transpose(2, 0, 1), 1).transpose((1, 2, 0))
        condition_mask = np.sum(c_ijt_copy, axis=1) < self.spine_num_per_pod * self.spine_up_port_num
        condition_mask = np.repeat(condition_mask[:, np.newaxis, :], self.pod_num, axis=1) & upper_triangular
        c_ijt_copy[condition_mask] += 1
        c_ijt_copy[condition_mask.transpose(1, 0, 2)] += 1

        # for t in range(self.spine_num_per_pod):
        #     for row_col_id in range(self.pod_num):
        #         for element_id in range(row_col_id + 1, self.pod_num):
        #             if np.sum(c_ijt[row_col_id, :, t]) < self.spine_num_per_pod * self.spine_up_port_num:
        #                 assert np.sum(c_ijt[row_col_id, :, t]) == np.sum(c_ijt[:, row_col_id, t])
        #                 c_ijt[row_col_id, element_id, t] += 1
        #                 c_ijt[element_id, row_col_id, t] += 1
        # assert np.all(c_ijt == c_ijt_copy)
        return c_ijt_copy

    def check_x_ijkt(self, x_ijkt: np.ndarray):
        pod_num = self.pod_num
        spine_num_per_pod = self.spine_num_per_pod
        spine_up_port_num = self.spine_up_port_num

        assert x_ijkt.shape == (pod_num, pod_num, spine_up_port_num, spine_num_per_pod), "x_ijkt shape does not match"
        x_ijt = np.sum(x_ijkt, axis=2)
        assert np.all(np.sum(x_ijt, axis=1) == spine_up_port_num), "spine's port should be fully used"

    def handle_failure_event(self, failure_id, duration_time):
        new_banned_gpu_status, new_banned_server_per_pod = self.random_fail_gpu(failure_id)
        if len(new_banned_gpu_status) > 0:
            self.failure_id_gpu_status_map[failure_id] = (new_banned_gpu_status, new_banned_server_per_pod)
            # 生成任务恢复事件
            from rapidAIsim.core.simulator import Simulator
            Simulator.register_event(
                RepairEvent(
                    duration_time,
                    failure_id,
                )
            )

    def random_fail_gpu(self, failure_id):
        # failure_id = -1*failure_id # gpus occupied by task i -1 is in failure
        new_banned_gpu_status, influenced_task_id, new_banned_server_per_pod = \
            self.gpu_placement_scheduler.random_fail_gpu(failure_id)
        if influenced_task_id != -1:
            # change remain model size of influenced_task
            from rapidAIsim.core.simulator import Simulator
            from rapidAIsim.task.waiting_task import WaitingTask
            Simulator.need_immediately_finish_task.append(influenced_task_id)
            handle_task_finish_immediately(influenced_task_id)
            # 任务相关流立即完成，同时任务再次进入等待队列
            task = Simulator.TASK_LIST[influenced_task_id]
            has_comp_ratio = Simulator.task_has_computation_time[influenced_task_id] / \
                                Simulator.task_need_comp_time[influenced_task_id]
            has_comm_ratio = Simulator.task_has_communication_size[influenced_task_id] / \
                                Simulator.task_need_comm_size[influenced_task_id]
            print(f'task {influenced_task_id} need comp {Simulator.task_need_comp_time[influenced_task_id]} '
                  f'need comm {Simulator.task_need_comm_size[influenced_task_id]}')
            print(f'task {influenced_task_id} has ratio {has_comp_ratio},{has_comm_ratio} '
                  f'has comp {Simulator.task_has_computation_time[influenced_task_id]} has comm '
                  f'{Simulator.task_has_communication_size[influenced_task_id]}')
            if has_comp_ratio > 1:
                print("debug has_comp_ratio", has_comp_ratio)
            assert has_comp_ratio <= 1.1
            if has_comm_ratio > 1:
                print("debug has_comp_ratio", has_comm_ratio)
            assert has_comm_ratio <= 1.1
            has_comp_ratio = min(has_comp_ratio, 0.9999)
            has_comm_ratio = min(has_comm_ratio, 0.9999)

            comm_time = task.duration_time - task.computation_time * task.computation_round
            remain_duration_time = (task.duration_time - has_comp_ratio * task.computation_time * task.computation_round
                                    - has_comm_ratio * comm_time)
            task.duration_time = remain_duration_time

            a_waiting_task = WaitingTask(Simulator.get_current_time(), task.model_size, task.gpu_num,
                                         Simulator.task_type_map[influenced_task_id], influenced_task_id, 0)
            Simulator.push_a_waiting_task(a_waiting_task)

        return new_banned_gpu_status, new_banned_server_per_pod

    def handle_repair_event(self, failure_id):
        self.gpu_placement_scheduler.repair_fail_gpu(self.failure_id_gpu_status_map[failure_id][0],
                                                     self.failure_id_gpu_status_map[failure_id][1])
