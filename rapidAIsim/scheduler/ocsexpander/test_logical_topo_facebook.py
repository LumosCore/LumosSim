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

        self.SCHEDULER_TIME_COST = {}
        self.init_mesh_topo()

    def schedule(self, strategy, taskid, alpha=1, beta=10):
        old_u = deepcopy(self.cur_c_ijt)
        c_ijt = deepcopy(self.cur_c_ijt)
        a_ijt = deepcopy(self.cur_c_ijt)
        base_time = time.time()
        self.random_change_mesh_topo(taskid)
        # with open(f'logical_topo_res/{arch}_{gpu_size}_logical_topo', 'a') as f:
        #         f.write(str(self.cosine_similarity(self.cur_c_ijt, old_u))+"\n")
                
        logical_topo_change_ratio = self.calculate_diff_ratio(old_u, self.cur_c_ijt)
        
        for t in range(self.spine_per_pod):
            tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(self.cur_c_ijt[:,:,t], self.pod_num)
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    if tmp_c_ijt_copy[i,j] == 0:
                        tmp_c_ijt_copy[i,j] = 1
            tmp_a_ijt,flag = TE_solver.solve(self.pod_num, self.spine_up_port_num//2, tmp_c_ijt_copy, np.zeros((self.pod_num, self.pod_num), dtype=int))
            c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
            a_ijt[:,:,t] = tmp_a_ijt

        base_time = time.time()
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
 
        if strategy != 'itv' and strategy != 'ilp_itv':
            from rapidAIsim.core.simulator import Simulator
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for t in range(self.spine_per_pod):
                        for k in range(self.spine_up_port_num):
                            x_ijkt[i,j,k,t] = min(x_ijkt[i,j,k,t], x_ijkt[j,i,k,t])
                
        time_cost = time.time() - base_time
        
        # 计算拓扑达成率
        # 对 x_ijk 在第三和第四个维度上求和得到新的二维数组
        sum_x_ijt = np.sum(x_ijkt, axis=(2))

        change_ratio = self.calculate_diff_ratio(x_ijkt, self.u_ijkt)
        
        self.u_ijkt = deepcopy(x_ijkt)
        cos_val = self.cosine_similarity(sum_x_ijt, c_ijt)

        return time_cost,cos_val,change_ratio,logical_topo_change_ratio
    
    # self.cosine_similarity(np.sum(sum_x_ijt, axis=-1), np.sum(c_ijt, axis=-1))
    
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
            # print("debug self.cur_c_ijt",self.cur_c_ijt.shape)
            self.cur_c_ijt[:,:,spine_id] = self.random_modify_links(self.cur_c_ijt[:,:,spine_id], self.spine_up_port_num,taskid)

        
    def cosine_similarity(self, a, b):
        # 展平为一维数组
        A = a.ravel()
        B = b.ravel()
        
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
    
    def calculate_diff_ratio(self,array1, array2):
        # 确保两个数组形状相同
        if array1.shape != array2.shape:
            raise ValueError("数组形状必须一致")
        
        # 逐元素比较，统计不相等的元素数量
        diff_mask = (array1 != array2)
        total_elements = array1.size
        diff_count = np.sum(diff_mask)
        
        # 计算差异比例
        diff_ratio = diff_count / total_elements
        return diff_ratio 

    def generate_ocs_configuration(self, c_ijt):
        u_ijkt = deepcopy(self.u_ijkt)
        # for t in range(self.spine_per_pod):
        #     tmp_u_ijt_copy, tmp_u_ijt_copy_T = divide_oxc_matrix.solve(self.cur_c_ijt[:,:,t], self.pod_num)
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_per_pod, oxc_list, 1,
                             c_ijt, self.u_a_ijkt, self.spine_up_port_num)
        x_ijkt, u_a_ijkt= m_solver.solve(True)
        self.u_a_ijkt = u_a_ijkt
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
    
    def generate_ocs_configuration_tradition(self, c_ijt):
        u_ijkt = self.u_ijkt
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_per_pod, oxc_list, 1,
                             c_ijt, u_ijkt, self.spine_up_port_num, True)
        x_ijkt = m_solver.solve()
        return x_ijkt

    def generate_ocs_configuration_ilp(self, c_ijt, alpha = 1, beta = 1):
        u_ijkt = self.u_ijkt
        x_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_per_pod),
                          dtype=int)
        
        flag, x_ijkt = mesh_solver_new.solve(self.spine_per_pod, self.spine_up_port_num, c_ijt, u_ijkt, False, True, only_facebook=True)
        if flag == False:
            return self.generate_ocs_configuration_bvn(c_ijt)
        return x_ijkt
    
    def generate_ocs_configuration_ilp_itv(self, a_ijt,c_ijt):
        u_ijkt = self.u_ijkt
        flag, x_star_ijkt = mesh_solver_new.solve(self.spine_per_pod, self.spine_up_port_num//2, a_ijt, u_ijkt, True, True, only_facebook=True)
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
    
    def test(self, strategy, c_ijt,  a_ijt):
        self.cur_c_ijt = c_ijt
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
 
        if strategy != 'itv' and strategy != 'ilp_itv':
            from rapidAIsim.core.simulator import Simulator
            for i in range(self.pod_num):
                for j in range(self.pod_num):
                    for t in range(self.spine_per_pod):
                        for k in range(self.spine_up_port_num):
                            x_ijkt[i,j,k,t] = min(x_ijkt[i,j,k,t], x_ijkt[j,i,k,t])
                
        sum_x_ijt = np.sum(x_ijkt, axis=(2))
        np.set_printoptions(threshold=np.inf)
        print("res c_ij of "+strategy+":")
        print(np.sum(sum_x_ijt, axis=-1))

        cos_val = self.cosine_similarity(sum_x_ijt, c_ijt)

        print("debug cos_val",cos_val, self.calculate_diff_ratio(sum_x_ijt, c_ijt))
def debug():
    spine_per_pod = 1
    spine_up_port_num = 8
    pod_num = 7
    

    # 填充链路需求
    cur_c_ijt = np.zeros((pod_num, pod_num, spine_per_pod), dtype=int)
    # 计算基础分配和剩余端口
    base = spine_up_port_num // (pod_num - 1)
    rem = spine_up_port_num % (pod_num - 1)
    # for i in range(pod_num // 2 - 1):
    #     for j in range(pod_num // 2 - 1):
    #         if i != j:
    #             cur_c_ijt[i, j, :] = spine_up_port_num // (pod_num // 2-1)*2
                
    # for i in range(pod_num // 2 + 1,pod_num):
    #     for j in range(pod_num // 2 + 1,pod_num):
    #         if i != j:
    #             cur_c_ijt[i, j, :] = spine_up_port_num // (pod_num // 2-1)*2
    
    for i in range(pod_num ):
        for j in range(pod_num):
            if i != j:
                cur_c_ijt[i, j, :] = spine_up_port_num // (pod_num - 1)
    print("debug rem",rem)
    # for i in range(rem-1):
    #     for j in range(i+1, rem):
    #         cur_c_ijt[i, j, :] += 1
    #         cur_c_ijt[j, i, :] += 1

    # for i in range(pod_num-1, max(-1, pod_num - rem +1 - 1), -1):
    #     for j in range(i-1, max(-1, pod_num - rem +1  - 2), -1):
    #         cur_c_ijt[i, j, :] += 1
    #         cur_c_ijt[j, i, :] += 1
            
    # for i in range(pod_num):
    #     j = pod_num - 1 - i  # 次对角线满足 i + j = pod_num - 1
    #     cur_c_ijt[i, j, 0] += 1


    print("origional c_ij:")
    print(np.sum(cur_c_ijt, axis=-1))
    remain_port = spine_up_port_num%(pod_num - 1)
    
    c_ijt = deepcopy(cur_c_ijt)
    a_ijt = deepcopy(cur_c_ijt)
    
    for t in range(spine_per_pod):
        tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(cur_c_ijt[:,:,t], pod_num)
        tmp_a_ijt,flag = TE_solver.solve(pod_num, spine_up_port_num//2, tmp_c_ijt_copy, tmp_c_ijt_copy, False)
        c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
        a_ijt[:,:,t] = tmp_a_ijt
    print("to calcualte logical topology c_ij:")
    print(c_ijt[:,:,0])
    gpu_size = pod_num*spine_up_port_num*spine_per_pod
    testOCS = testL2OCS(gpu_size, pod_num, spine_per_pod, spine_up_port_num)

    testOCS.test('itv',c_ijt, a_ijt)
    testOCS.test('ilp',c_ijt, a_ijt)
    testOCS.test('ilp_itv',c_ijt, a_ijt)

def debug2():
    spine_per_pod = 1
    spine_up_port_num = 16
    pod_num = 8
    

    # 填充链路需求
    cur_c_ijt = np.zeros((pod_num, pod_num, spine_per_pod), dtype=int)
    # 计算基础分配端口
    # for i in range(pod_num ):
    #     for j in range(pod_num):
    #         if i != j:
    #             cur_c_ijt[i, j, :] = spine_up_port_num // (pod_num - 1)

    # cur_c_ijt[:,:,0] = [[0,4,4,0],[4,0,4,0],[4,4,0,0],[0,0,0,0]]
    cur_c_ijt[:,:,0] = [[0,2,2,2,2,0,0,0],[2,0,2,2,2,0,0,0],[2,2,0,2,2,0,0,0],[2,2,2,0,2,0,0,0],[2,2,2,2,0,0,0,0],[0,0,0,0,0,0,4,4],[0,0,0,0,0,4,0,4],[0,0,0,0,0,4,4,0]
    ]


    print("origional c_ij:")
    print(np.sum(cur_c_ijt, axis=-1))
    remain_port = spine_up_port_num%(pod_num - 1)
    
    c_ijt = deepcopy(cur_c_ijt)
    a_ijt = deepcopy(cur_c_ijt)
    
    # for t in range(spine_per_pod):
    #     tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(cur_c_ijt[:,:,t], pod_num)
    #     tmp_a_ijt,flag = TE_solver.solve(pod_num, spine_up_port_num//2, tmp_c_ijt_copy, tmp_c_ijt_copy, False)
    #     c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
    #     a_ijt[:,:,t] = tmp_a_ijt
    c_ijt = c_ijt
    a_ijt = c_ijt

    print("to calcualte logical topology c_ij:")
    print(c_ijt[:,:,0])
    gpu_size = pod_num*spine_up_port_num*spine_per_pod
    testOCS = testL2OCS(gpu_size, pod_num, spine_per_pod, spine_up_port_num)

    testOCS.test('itv',c_ijt, a_ijt)
    testOCS.test('ilp',c_ijt, a_ijt)
    testOCS.test('ilp_itv',c_ijt, a_ijt)
    # testOCS.test('bvn',c_ijt, a_ijt)
    # testOCS.test('helios',c_ijt, a_ijt)
    
    
if __name__ == '__main__':
    # spine_per_pod = 1
    # spine_up_port_num = 16
    # pod_num = 8
    

    # # 填充链路需求
    # cur_c_ijt = np.zeros((pod_num, pod_num, spine_per_pod), dtype=int)
    # # 计算基础分配端口
    # # for i in range(pod_num ):
    # #     for j in range(pod_num):
    # #         if i != j:
    # #             cur_c_ijt[i, j, :] = spine_up_port_num // (pod_num - 1)

    # # cur_c_ijt[:,:,0] = [[0,4,4,0],[4,0,4,0],[4,4,0,0],[0,0,0,0]]
    # cur_c_ijt[:,:,0] = [[0,8,8,0,0,0,0,0],
    #                     [8,0,8,0,0,0,0,0],
    #                     [8,8,0,0,0,0,0,0],
    #                     [0,0,0,0,0,0,0,0],
    #                     [0,0,0,0,0,0,0,0],
    #                     [0,0,0,0,0,0,0,0],
    #                     [0,0,0,0,0,0,0,16],
    #                     [0,0,0,0,0,0,16,0]
    # ]
    # print("origional c_ij:")
    # print(np.sum(cur_c_ijt, axis=-1))
    
    # c_ijt = deepcopy(cur_c_ijt)
    # a_ijt = deepcopy(cur_c_ijt)
    
    # for t in range(spine_per_pod):
    #     tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(cur_c_ijt[:,:,t], pod_num)
    #     tmp_a_ijt,flag = TE_solver.solve(pod_num, spine_up_port_num//2, tmp_c_ijt_copy, tmp_c_ijt_copy, False)
    #     c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
    #     a_ijt[:,:,t] = tmp_a_ijt
    # print("to calcualte logical topology c_ij:")
    # print(c_ijt[:,:,0])
    # gpu_size = pod_num*spine_up_port_num*spine_per_pod
    # testOCS = testL2OCS(gpu_size, pod_num, spine_per_pod, spine_up_port_num)

    # # testOCS.test('itv',c_ijt, a_ijt)
    # testOCS.test('ilp',c_ijt, a_ijt)
    # # testOCS.test('ilp_itv',c_ijt, a_ijt)
    
    
    spine_per_pod = 1
    spine_up_port_num = 16
    pod_num = 3
    

    # 填充链路需求
    cur_c_ijt = np.zeros((pod_num, pod_num, spine_per_pod), dtype=int)
    # 计算基础分配和剩余端口
    base = spine_up_port_num // (pod_num - 1)
    rem = spine_up_port_num % (pod_num - 1)

    
    for i in range(pod_num):
        for j in range(pod_num):
            if i != j:
                cur_c_ijt[i, j, :] = spine_up_port_num // (pod_num - 1)
    print("debug rem",rem)


    print("origional c_ij:")
    print(np.sum(cur_c_ijt, axis=-1))
    remain_port = spine_up_port_num%(pod_num - 1)
    
    c_ijt = deepcopy(cur_c_ijt)
    a_ijt = deepcopy(cur_c_ijt)
    
    for t in range(spine_per_pod):
        tmp_c_ijt_copy, tmp_c_ijt_copy_T = divide_oxc_matrix.solve(cur_c_ijt[:,:,t], pod_num)
        tmp_a_ijt,flag = TE_solver.solve(pod_num, spine_up_port_num//2, tmp_c_ijt_copy, tmp_c_ijt_copy, False)
        c_ijt[:,:,t] = tmp_a_ijt + tmp_a_ijt.T
        a_ijt[:,:,t] = tmp_a_ijt
    print("to calcualte logical topology c_ij:")
    print(c_ijt[:,:,0])
    gpu_size = pod_num*spine_up_port_num*spine_per_pod
    testOCS = testL2OCS(gpu_size, pod_num, spine_per_pod, spine_up_port_num)


    testOCS.test('ilp',c_ijt, a_ijt)

