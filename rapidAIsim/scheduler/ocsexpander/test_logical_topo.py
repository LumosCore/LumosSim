

import time
import logging
from collections import defaultdict
import numpy as np
import os
import warnings
from copy import deepcopy
import sys

from rapidAIsim.scheduler.ocsexpander.cijt_solver import CijtSolver
from rapidAIsim.scheduler.ocsexpander.mcf_solver import MCFSolver
from rapidAIsim.scheduler.ocsexpander.greedy_solver import GreedySolver
from rapidAIsim.scheduler.ocsexpander.routing_solver import RoutingSolver
from rapidAIsim.scheduler.ocsexpander import mesh_solver_new, TE_solver_lp, TE_solver, gpu_placement, divide_oxc_matrix,divide_oxc_matrix_mcf
from rapidAIsim.communication_strategy.all2all import All2All
from rapidAIsim.communication_strategy.ring import Ring
from rapidAIsim.core.event.RepairEvent import RepairEvent
from rapidAIsim.core.network_refresh import handle_task_finish_immediately
from rapidAIsim.scheduler.ocsexpander.helios import bg
from rapidAIsim.scheduler.ocsexpander.b import NetworkModifier

logging.basicConfig(level=logging.ERROR)


class testL2OCS:
    def __init__(self, gpu_size, pod_num, spine_per_pod, spine_up_port_num):
        assert gpu_size == pod_num*spine_per_pod*spine_up_port_num
        self.gpu_size = gpu_size
        self.pod_num = pod_num
        self.spine_per_pod = spine_per_pod
        self.spine_up_port_num = spine_up_port_num
        self.ocs_num = spine_up_port_num*spine_per_pod
       
        self.x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),dtype=float)
        self.u_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),dtype=float)+0.000001
        self.u_a_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num//2, self.spine_per_pod),dtype=float)+0.000001
        
        self.cur_c_ijt = np.zeros((self.pod_num, self.pod_num, self.spine_per_pod), dtype=int)

        self.spine_nm_map = {}
        self.spine_clique_map = {}
        # 初始化nm
        for spine_id in range(spine_per_pod):
            self.spine_nm_map[spine_id] = NetworkModifier(pod_num)
        # 初始化clique
        for spine_id in range(spine_per_pod):
            clique = self.spine_nm_map[spine_id].next_topology(None, self.spine_up_port_num, spine_id)
            self.spine_clique_map[spine_id] = clique
        # 初始化逻辑拓扑
        for spine_id in range(spine_per_pod):
            clique = self.spine_clique_map[spine_id]
            self.cur_c_ijt[:,:,spine_id] = self.spine_nm_map[spine_id].clique_to_cij(clique)
        # print("debug init logical topo")
        # print(self.cur_c_ijt[:,:,0])
        
        # self.init_mesh_topo()

    def schedule(self, strategy, taskid, alpha=1, beta=10):
        old_u = deepcopy(self.cur_c_ijt)
        c_ijt = deepcopy(self.cur_c_ijt)
        a_ijt = deepcopy(self.cur_c_ijt)
        base_time = time.time()
        # 更新clique
        for spine_id in range(spine_per_pod):
            clique = self.spine_clique_map[spine_id]
            clique = self.spine_nm_map[spine_id].next_topology(clique, self.spine_up_port_num, spine_id*2)
            self.spine_clique_map[spine_id] = clique
        # 更新逻辑拓扑
        for spine_id in range(self.spine_per_pod):
            clique = self.spine_clique_map[spine_id]
            self.cur_c_ijt[:,:,spine_id] = self.spine_nm_map[spine_id].clique_to_cij(clique)
        # print("debug new logical topo")
        # print(self.cur_c_ijt[:,:,0])
        # print(self.spine_clique_map[0])

        logical_topo_change_ratio = self.calculate_diff_ratio(old_u, self.cur_c_ijt, self.spine_up_port_num)
        
        for t in range(self.spine_per_pod):
            tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(self.cur_c_ijt[:,:,t], self.pod_num)
            # for i in range(self.pod_num):
            #     for j in range(self.pod_num):
            #         if tmp_c_ijt_copy[i,j] == 0:
            #             tmp_c_ijt_copy[i,j] = 1
            tmp_a_ijt,flag = TE_solver.solve(self.pod_num, self.spine_up_port_num//2, tmp_c_ijt_copy, np.zeros((self.pod_num, self.pod_num), dtype=int))
            c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
            a_ijt[:,:,t] = tmp_a_ijt
            
        # print("debug new logical topo after cal")
        # print(c_ijt[:,:,0])
        # print(self.spine_clique_map[0])

        base_time = time.time()
        if strategy == 'bvn':
            x_ijkt = self.generate_ocs_configuration_bvn(c_ijt)
        elif strategy == 'ilp':
            x_ijkt = self.generate_ocs_configuration_ilp(c_ijt)
        elif strategy == 'helios':
            x_ijkt = self.generate_helios_configuration(c_ijt)
        elif strategy == 'bvn_itv':
            x_ijkt = self.generate_ocs_configuration_bvn_itv(a_ijt)
        elif strategy == 'ilp_itv':
            x_ijkt = self.generate_ocs_configuration_ilp_itv(a_ijt,c_ijt)
        elif strategy == 'itv':
            x_ijkt = self.generate_ocs_configuration(a_ijt)
        else:
            assert False
 

        if strategy not in {'itv', 'ilp_itv', 'bvn_itv'}:
            from rapidAIsim.core.simulator import Simulator
            for t in range(self.spine_per_pod):
                for k in range(self.spine_up_port_num):
                    for i in range(self.pod_num):
                        for j in range(i, self.pod_num):
                            if x_ijkt[i, j, k, t] != x_ijkt[j, i, k, t]:
                                val1 = x_ijkt[i, j, k, t]
                                val2 = x_ijkt[j, i, k, t]
                                if strategy in {'bvn', 'helios'} and val1 < val2:
                                    val = val2
                                    sum_cond = [
                                        np.sum(x_ijkt[i, :, :, t]) + val <= self.spine_up_port_num,
                                        np.sum(x_ijkt[:, j, :, t]) + val <= self.spine_up_port_num,
                                        np.sum(x_ijkt[j, :, :, t]) + val <= self.spine_up_port_num,
                                        np.sum(x_ijkt[:, i, :, t]) + val <= self.spine_up_port_num
                                    ]
                                    if all(sum_cond):
                                        x_ijkt[i, j, k, t] = max(val1, val2)
                                    else:
                                        x_ijkt[i, j, k, t] = min(val1, val2)
                                else:
                                    x_ijkt[i, j, k, t] = min(val1, val2)
                
        time_cost = time.time() - base_time
        
        # 计算拓扑达成率
        # 对 x_ijk 在第三和第四个维度上求和得到新的二维数组
        sum_x_ijt = np.sum(x_ijkt, axis=(2))

        change_ratio = self.calculate_diff_ratio(self.u_ijkt, x_ijkt, 1)
        
        self.u_ijkt = deepcopy(x_ijkt)
        cos_val = self.calculate_diff_ratio(sum_x_ijt, c_ijt, self.spine_up_port_num)

        return time_cost,cos_val,change_ratio,logical_topo_change_ratio
    

    def random_modify_links(self, c_ij, u, seed = None, times = 10):
      if seed is not None:
        np.random.seed(seed)
      deg = [sum(c_ij[i]) for i in range(self.pod_num)]
      edges = sum([sum([c_ij[i][j] for j in range(self.pod_num)]) for i in range(self.pod_num)])
      for _ in range(times):
        opt = np.random.randint(0,2)
        # print("cur logi topo")
        # np.set_printoptions(threshold=np.inf)
        # print(c_ij)
        if opt == 0:
          # Add a link
          i = j = 0
          # print("Trying to add a link")
          if edges <= 0.5 * self.pod_num * u:
            # while i == j or deg[i] >= u or deg[j] >= u:
            #   print("debug edges:",edges,0.5 * self.pod_num * (self.pod_num-1) / 2 * u, self.pod_num)
            #   print("debug deg[i]",u,i,j,deg[i],deg[j])
            #   i, j = np.random.randint(0, self.pod_num, size=2)
            while i == j or deg[i] >= u or deg[j] >= u:
              i, j = np.random.randint(0, self.pod_num, size=2)
          if i == j or deg[i] >= u or deg[j] >= u:
            continue
          # print("Add link between {} and {}".format(i, j))
          c_ij[i][j] += 1
          c_ij[j][i] += 1
          deg[i] += 1
          deg[j] += 1
          edges += 2
        else:
          # Remove a link
          i = j = 0
          # print("Trying to remove a link")
          if edges >= 0.5 * self.pod_num * u:
            while i == j or c_ij[i][j] <= 0:
              i, j = np.random.randint(0, self.pod_num, size=2)
          if i == j or c_ij[i][j] <= 0:
            continue
          # print("Remove link between {} and {}".format(i, j))
          c_ij[i][j] -= 1
          c_ij[j][i] -= 1
          deg[i] -= 1
          deg[j] -= 1
          edges -= 2
      return c_ij

        

    def init_mesh_topo(self):
        for i in range(0, self.pod_num, 3):
            for t in range(self.spine_per_pod):
                if i+1 < self.pod_num:
                    self.cur_c_ijt[i,i+1,t] = self.spine_up_port_num/2
                    self.cur_c_ijt[i+1,i,t] = self.spine_up_port_num/2
                if i+2 < self.pod_num:
                    self.cur_c_ijt[i,i+2,t] = self.spine_up_port_num/2
                    self.cur_c_ijt[i+2,i,t] = self.spine_up_port_num/2
                    self.cur_c_ijt[i+1,i+2,t] = self.spine_up_port_num/2
                    self.cur_c_ijt[i+2,i+1,t] = self.spine_up_port_num/2

        
    def random_change_mesh_topo(self,taskid):
        for spine_id in range(self.spine_per_pod):


            self.cur_c_ijt[:,:,spine_id] = self.random_modify_links(self.cur_c_ijt[:,:,spine_id], self.spine_up_port_num,taskid)

        
    def cosine_similarity(self, a, b):
        dot_product = np.dot(a.flatten(), b.flatten())
        norm_a = np.linalg.norm(a.flatten())
        norm_b = np.linalg.norm(b.flatten())
        return dot_product / (norm_a * norm_b)
    

    import numpy as np

    def calculate_diff_ratio(self,array1, array2, k_east):
        # 展平为一维数组
        A = array1.ravel()
        B = array2.ravel()
        
        # 计算满足条件的差值之和
        diff = np.where(A < B, B - A, 0)
        sum_diff = np.sum(diff)
        
        # 计算 array2 的总和
        sum_B = np.sum(B)
        
        # 处理 array2 总和为 0 的情况
        if sum_B == 0:
            return 0.0
        
        # 返回比例
        return 1-sum_diff / sum_B
        # # 展平为一维数组
        # A = array1.ravel()
        # B = array2.ravel()
        
        # # 计算点积
        # dot_product = np.dot(A, B)
        
        # # 计算模长
        # norm_A = np.linalg.norm(A)
        # norm_B = np.linalg.norm(B)
        
        # # 处理零向量
        # if norm_A == 0 or norm_B == 0:
        #     return 0.0
        
        # # 原始余弦相似度
        # cosine_sim = dot_product / (norm_A * norm_B)
        
        # # 长度比例因子
        # length_ratio = min(norm_A, norm_B) / max(norm_A, norm_B)
        
        # # 最终相似度
        # return cosine_sim * length_ratio


    def generate_ocs_configuration(self, c_ijt):
        u_ijkt = deepcopy(self.u_ijkt)
        # for t in range(self.spine_per_pod):
        #     tmp_u_ijt_copy, tmp_u_ijt_copy_T = divide_oxc_matrix.solve(self.cur_c_ijt[:,:,t], self.pod_num)
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_per_pod, oxc_list, 1,
                             c_ijt, self.u_a_ijkt, self.spine_up_port_num)
        x_ijkt, u_a_ijkt= m_solver.solve(True)
        self.u_a_ijkt = x_ijkt
        return x_ijkt
    
    

    
    def generate_ocs_configuration_ilp_itv(self, a_ijt,c_ijt):
        u_ijkt = self.u_ijkt
        flag, x_star_ijkt = mesh_solver_new.solve(self.spine_per_pod, self.spine_up_port_num//2, a_ijt, self.u_a_ijkt, True)
        self.u_a_ijkt = x_star_ijkt
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),
                          dtype=int)
        if flag == False:
            return self.generate_ocs_configuration_bvn_itv(a_ijt)
        else:
            for k in range(0, self.spine_up_port_num // 2, 1):
                x_ijkt[:, :, 2 * k, :] += x_star_ijkt[:, :, k, :]
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for k in range(0, self.spine_up_port_num, 2):
                        x_ijkt[j, i, k + 1, :] = x_ijkt[i, j, k, :]
        return x_ijkt
    
    def generate_ocs_configuration_bvn_itv(self, a_ijt):
        c_ijt = a_ijt
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),
                          dtype=int)
        import tensorflow as tf
        import bvn
        for t in range(self.spine_per_pod):
            new_array = c_ijt[:, :, t]
            new_array_tf = tf.expand_dims(tf.constant(new_array, dtype=tf.float32, name='doubly_stochastic'),axis=0)
            p, c = bvn.bvn(new_array_tf, self.spine_up_port_num//2)
            matrix = p[0]
            coff = c[0]
            res_array_np = self.generate_res_array(coff, matrix, self.spine_up_port_num//2)
            for k in range(0, self.spine_up_port_num//2, 1):
                x_ijkt[:, :, 2 * k, t] += res_array_np[:, :, k]
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for k in range(0, self.spine_up_port_num, 2):
                        x_ijkt[j, i, k + 1, t] = x_ijkt[i, j, k, t]
        return x_ijkt
    
    def generate_helios_configuration(self, c_ijt):
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),
                          dtype=int)
        for t in range(self.spine_per_pod):
            d_wave = c_ijt[:,:,t]
            base_capacity = 1
            r = self.spine_up_port_num
            x_ijkt[:,:,:,t] = bg(r, d_wave, base_capacity)
        return x_ijkt
    
    def generate_ocs_configuration_ilp(self, c_ijt, alpha = 1, beta = 1):
        u_ijkt = self.u_ijkt
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),
                          dtype=int)
        
        flag, x_ijkt = mesh_solver_new.solve(self.spine_per_pod, self.spine_up_port_num, c_ijt, u_ijkt, False, True)
        if flag == False:
            return self.generate_ocs_configuration_bvn(c_ijt)
        return x_ijkt
    
    def generate_ocs_configuration_bvn(self, c_ijt):
        u_ijkt = self.u_ijkt
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),
                          dtype=int)
        import tensorflow as tf
        import bvn
        for t in range(self.spine_per_pod):
            new_array = c_ijt[:, :, t]
            new_array_tf = tf.expand_dims(tf.constant(new_array, dtype=tf.float32, name='doubly_stochastic'),axis=0)
            p, c = bvn.bvn(new_array_tf, self.spine_up_port_num)
            matrix = p[0]
            coff = c[0]
            res_array_np = self.generate_res_array(coff, matrix, self.spine_up_port_num)
            x_ijkt[:, :, :, t] += res_array_np
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

        
        return res_array_np
    
    def test(self, strategy, c_ijt):
        self.cur_c_ijt = c_ijt
        a_ijt = deepcopy(self.cur_c_ijt)
        for t in range(self.spine_per_pod):
            tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(self.cur_c_ijt[:,:,t], self.pod_num)
            # for i in range(self.pod_num):
            #     for j in range(self.pod_num):
            #         if tmp_c_ijt_copy[i,j] == 0:
            #             tmp_c_ijt_copy[i,j] = 1
            tmp_a_ijt,flag = TE_solver.solve(self.pod_num, self.spine_up_port_num//2, tmp_c_ijt_copy, np.zeros((self.pod_num, self.pod_num), dtype=int))
            c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
            a_ijt[:,:,t] = tmp_a_ijt
        if strategy == 'bvn':
            x_ijkt = self.generate_ocs_configuration_bvn(c_ijt)
        elif strategy == 'ilp':
            x_ijkt = self.generate_ocs_configuration_ilp(c_ijt)
        elif strategy == 'helios':
            x_ijkt = self.generate_helios_configuration(c_ijt)
        elif strategy == 'ilp_itv':
            x_ijkt = self.generate_ocs_configuration_ilp_itv(a_ijt,c_ijt)
        elif strategy == 'itv':
            x_ijkt = self.generate_ocs_configuration(a_ijt)
        else:
            assert False
            
        for i in range(self.pod_num):
            for t in range(self.spine_per_pod):
                for k in range(self.spine_up_port_num):
                    assert np.sum(x_ijkt[:,i,k,t]) <= 1
                    assert np.sum(x_ijkt[i,:,k,t]) <= 1
                            
 
        if strategy != 'itv' and strategy != 'ilp_itv':
            from rapidAIsim.core.simulator import Simulator
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for t in range(self.spine_per_pod):
                        for k in range(self.spine_up_port_num):
                            x_ijkt[i,j,k,t] = min(x_ijkt[i,j,k,t], x_ijkt[j,i,k,t])
                
        sum_x_ijt = np.sum(x_ijkt, axis=(2))
        cos_val = self.calculate_diff_ratio(sum_x_ijt, c_ijt, self.spine_up_port_num)

        return time_cost,cos_val,change_ratio,logical_topo_change_ratio

    
if __name__ == '__main__':
    total_task = 100

    arch = str(sys.argv[1])
    gpu_size = int(sys.argv[2])
    print("test ",arch)
    with open(f'logical_topo_res/{arch}_{gpu_size}_logical_topo_result', 'w') as f:
        f.write(str(("task_id","time_cost","LTCR","cos"))+"\n")
    # with open(f'logical_topo_res/{arch}_{gpu_size}_logical_topo', 'w') as f:
    #     f.write(str(("start"))+"\n")
    spine_per_pod = 16
    spine_up_port_num = 16
    pod_num = gpu_size // (spine_per_pod*spine_up_port_num)
    testOCS = testL2OCS(gpu_size, pod_num, spine_per_pod, spine_up_port_num)
    cos_list = []
    time_list = []
    change_list = []
    logical_topo_change_list = []
    for i in range(total_task):
        time_cost,cos,change,logical_topo_change_ratio = testOCS.schedule(arch,i)
        with open(f'logical_topo_res/{arch}_{gpu_size}_logical_topo_result', 'a') as f:
            f.write(str((i,time_cost,cos,change,logical_topo_change_ratio))+"\n")
        cos_list.append(float(cos))
        time_list.append(time_cost)
        change_list.append(float(change))
        logical_topo_change_list.append(float(logical_topo_change_ratio))
    cos_list.sort()
    time_list.sort()
    change_list.sort()
    logical_topo_change_list.sort()
    print("time cost")
    print(sum(time_list)/(total_task))
    print("LTRL ")
    print(sum(cos_list)/(total_task))
    print("MA ")
    print(sum(change_list[:-1])/(total_task-1))
    print("Logical Topo Change Ratio ")
    print(sum(logical_topo_change_list[:-1])/(total_task-1))
    print(time_list)
    print(cos_list)
    print(change_list)
    print(logical_topo_change_list)


    
    
    print('----------------------------------------')
    print('----------------------------------------')
    print('----------------------------------------')
    print('----------------------------------------')

