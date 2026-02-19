# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，这一阶段不涉及connection manager
# 2. 当需要跨leaf通信时，此时有两种情况：

import numpy as np
import math


class ConnectionManager:
    def __init__(self, gpu_num=512, server_num=64, leaf_num=16, spine_num=16, inference_model=None):
        self.gpu_num = gpu_num
        self.server_num = server_num
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.server_per_leaf = int(server_num / leaf_num)
        self.gpu_per_server = int(gpu_num / server_num)
        if inference_model is None:
            raise ValueError("inference model is None. Please check the model path.")
        self.model = inference_model

        self.leaf_to_spine_remain_port_num = {}
        for leaf_id in range(leaf_num):
            for to_spine_id in range(spine_num):
                if leaf_id not in self.leaf_to_spine_remain_port_num:
                    self.leaf_to_spine_remain_port_num[leaf_id] = {}
                self.leaf_to_spine_remain_port_num[leaf_id][to_spine_id] = 1

    def find_valid_ai_iter(self, require_gpu_size, leaf_remain_empt_server_list, start_require_leaf_num,
                           leaf_remain_empt_gpu_list, link_weight):
        cur_loss = 10000000
        c_i_j_solution = None
        x_i_solution = None
        final_require_spine_num = -1
        final_leaf_valid_vector = []

        require_leaf_num = start_require_leaf_num
        while require_leaf_num < require_gpu_size and require_leaf_num <= self.leaf_num and require_leaf_num < int(
                require_gpu_size / self.gpu_per_server):
            require_spine_num = int(require_gpu_size / require_leaf_num)
            leaf_valid_vector = []
            potentional_leaf_list = []
            # Step1. 在leaf_resource_manager中选取合适的leafgroup
            for temp_leaf_id in range(self.leaf_num):
                require_server_num = math.ceil(require_spine_num / self.gpu_per_server)
                if leaf_remain_empt_server_list[temp_leaf_id] >= require_server_num:
                    potentional_leaf_list.append([temp_leaf_id])
                    # leaf_valid_vector.append(require_server_num/leaf_remain_empt_server_list[temp_leaf_id])
                    leaf_valid_vector.append(1)
                else:
                    leaf_valid_vector.append(0)
            if len(potentional_leaf_list) < require_leaf_num:
                require_leaf_num *= 2
            else:
                original_state = np.zeros((self.leaf_num, self.spine_num))
                for leaf_id in range(self.leaf_num):
                    for spine_id in range(self.spine_num):
                        # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]<0:
                        #     original_state[leaf_id, spine_id] = self.leaf_to_spine_remain_port_num[leaf_id][spine_id]*(1+0.1*link_weight[str(leaf_id) + '_' + str(spine_id)])
                        # else:
                        original_state[leaf_id, spine_id] = self.leaf_to_spine_remain_port_num[leaf_id][spine_id]
                link_weight_numpy = np.zeros((self.leaf_num, self.spine_num))
                for leaf_id in range(self.leaf_num):
                    for spine_id in range(self.spine_num):
                        if str(leaf_id) + '_' + str(spine_id) in link_weight:
                            link_weight_numpy[leaf_id, spine_id] = link_weight[str(leaf_id) + '_' + str(spine_id)]
                cur_c_i_j_solution, cur_x_i_solution, _ = self.model.inference(
                    [require_leaf_num, require_spine_num], original_state, leaf_valid_vector, link_weight_numpy)
                f1 = open('problems.csv', 'a')
                line = np.concatenate((
                    np.array([require_leaf_num, require_spine_num], dtype=np.float32).flatten(),
                    np.array(leaf_valid_vector, dtype=np.float32).flatten(),
                    np.array(original_state, dtype=np.float32).flatten(),
                    np.array(link_weight_numpy / 256, dtype=np.float32).flatten()
                ))
                line = [str(i) for i in line]
                f1.write(','.join(line))
                f1.write('\n')
                # f1.write(str(require_leaf_num))
                # f1.write(",")
                # f1.write(str(require_spine_num))
                # f1.write("\n")
                # f1.close()
                # f1 = open('original_state.txt', 'a')
                # f1.write(str(original_state))
                # f1.write("\n")
                # f1.close()
                # f1 = open('leaf_valid_vector.txt', 'a')
                # f1.write(str(leaf_valid_vector))
                # f1.write("\n")
                # f1.close()
                # f1 = open('link_weight_numpy.txt', 'a')
                # f1.write(str(link_weight_numpy))
                # f1.write("\n")
                f1.close()
                tmp_cur_loss = 0
                for leaf_id in range(self.leaf_num):
                    for spine_id in range(self.spine_num):
                        if cur_c_i_j_solution[str(leaf_id) + '_' + str(spine_id)] > 0 and str(leaf_id) + '_' + str(
                                spine_id) in link_weight:
                            tmp_cur_loss += cur_c_i_j_solution[str(leaf_id) + '_' + str(spine_id)] * link_weight[
                                str(leaf_id) + '_' + str(spine_id)]
                tmp_cur_loss /= (1 + require_leaf_num / self.leaf_num)
                for k, v in cur_x_i_solution.items():
                    if v > 0:
                        if leaf_valid_vector[int(k)] != 1:
                            print("debug leaf_valid_vector", leaf_valid_vector, k, v)
                            print(cur_x_i_solution)
                        assert leaf_valid_vector[int(k)] > 0
                if tmp_cur_loss < cur_loss:
                    cur_loss = tmp_cur_loss
                    c_i_j_solution = cur_c_i_j_solution
                    x_i_solution = cur_x_i_solution
                    final_require_spine_num = require_spine_num
                    final_leaf_valid_vector = leaf_valid_vector
                require_leaf_num *= 2

        if cur_loss == 10000000 or cur_loss > 1024:
            return False, None, None, None, None

        if cur_loss > 0:
            print("debug x_i_solution_con", cur_loss)
        else:
            print("debug x_i_solution_no", cur_loss)
        for k, v in x_i_solution.items():
            if v > 0:
                print(k, v, final_leaf_valid_vector[int(k)])
        leaf_occupy_gpu_num_map = {}
        allocation_link_mapping = []
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                if round(c_i_j_solution[str(leaf_id) + '_' + str(spine_id)]) > 0:
                    # assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>0
                    self.leaf_to_spine_remain_port_num[leaf_id][spine_id] -= round(
                        c_i_j_solution[str(leaf_id) + '_' + str(spine_id)])
                    allocation_link_mapping.append([self.gpu_num + leaf_id, self.gpu_num + self.leaf_num + spine_id,
                                                    round(c_i_j_solution[str(leaf_id) + '_' + str(spine_id)])])
                    allocation_link_mapping.append([self.gpu_num + self.leaf_num + spine_id, self.gpu_num + leaf_id,
                                                    round(c_i_j_solution[str(leaf_id) + '_' + str(spine_id)])])
                    if str(leaf_id) + '_' + str(spine_id) not in link_weight:
                        link_weight[str(leaf_id) + '_' + str(spine_id)] = 0
                    link_weight[str(leaf_id) + '_' + str(spine_id)] += require_gpu_size
        for leaf_id in range(self.leaf_num):
            if round(x_i_solution[str(leaf_id)]):
                leaf_occupy_gpu_num_map[leaf_id] = round(x_i_solution[str(leaf_id)] * final_require_spine_num)
        job_allocated_leaf_spine_link = {}
        for leaf_id in range(self.leaf_num):
            if leaf_id not in job_allocated_leaf_spine_link:
                job_allocated_leaf_spine_link[leaf_id] = {}
            for spine_id in range(self.spine_num):
                if spine_id not in job_allocated_leaf_spine_link[leaf_id]:
                    job_allocated_leaf_spine_link[leaf_id][spine_id] = 0
                job_allocated_leaf_spine_link[leaf_id][spine_id] += round(
                    c_i_j_solution[str(leaf_id) + '_' + str(spine_id)])
                # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>=0:
        may_contention_link = {}
        link_contention_res = 0
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                if self.leaf_to_spine_remain_port_num[leaf_id][spine_id] < 0:
                    link_contention_res += 1
                    if f'{leaf_id}_{spine_id}' not in may_contention_link:
                        may_contention_link[f'{leaf_id}_{spine_id}'] = 0
                    may_contention_link[f'{leaf_id}_{spine_id}'] += 1
        if link_contention_res > 0:
            print("return contention res:", link_contention_res, cur_loss)
        leaf_remain_gpu_num_map = None
        return True, allocation_link_mapping, leaf_occupy_gpu_num_map, job_allocated_leaf_spine_link, may_contention_link

    def print_connection_info(self):
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id] >0:
                print(leaf_id, spine_id, self.leaf_to_spine_remain_port_num[leaf_id][spine_id])

    def release_connection_resource(self, job_allocated_leaf_spine_link):
        for leaf_id in job_allocated_leaf_spine_link:
            for spine_id in job_allocated_leaf_spine_link[leaf_id]:
                self.leaf_to_spine_remain_port_num[leaf_id][spine_id] += job_allocated_leaf_spine_link[leaf_id][
                    spine_id]

# connection_manager_ = ConnectionManager(512, 64, 32, 15)
# connection_manager_.find_valid_gpu_for_no_pow2_tas_releax()
