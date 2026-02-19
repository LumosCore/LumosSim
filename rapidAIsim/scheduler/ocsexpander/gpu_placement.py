from copy import deepcopy
import numpy as np
import gurobipy
import math
import random


class GPUPlacement:
    def __init__(self):
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        self.busy_server_per_pod = np.zeros(infra.pod_num, dtype=int)
        self.pod_num = infra.pod_num
        self.server_num_per_pod = infra.server_num_per_pod
        self.task_occupy_resource_map = {}
        self.task_occupy_gpu_map = {}
        self.leaf_per_pod = infra.leaf_num_per_pod
        self.TP = infra.NIC_num_in_a_server
        self.gpu_per_pod = self.server_num_per_pod * self.TP
        self.pod_gpu_status = np.zeros((infra.pod_num, self.leaf_per_pod, infra.server_per_leaf * self.TP), dtype=int)
        self.server_per_leaf = infra.server_per_leaf
        self.banned_gpu_status = np.zeros((infra.pod_num, self.leaf_per_pod, infra.server_per_leaf * self.TP), dtype=int) #记录gpu是否被故障
        self.banned_server_per_pod = np.zeros(infra.pod_num, dtype=int)

    def occupy_gpu(self, pod_id, used_server_num, task_id):
        potential_leaf_pair_list = []
        for tmp_leaf_id in range(self.leaf_per_pod):
            tmp_leaf_remain_server_num = self.server_per_leaf - np.sum(
                self.pod_gpu_status[pod_id, tmp_leaf_id, :]) // self.TP - np.sum(self.banned_gpu_status[pod_id, tmp_leaf_id, :]) // self.TP
            potential_leaf_pair_list.append((tmp_leaf_id, tmp_leaf_remain_server_num))

        model = gurobipy.Model("PodStrategy solution")
        model.setParam('OutputFlag', 0)
        #model.setParam('TimeLimit', 300)
        x_i = {}
        used_i = {}

        for tmp_leaf_id in range(self.leaf_per_pod):
            x_i[tmp_leaf_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.server_per_leaf - np.sum(
                self.pod_gpu_status[pod_id, tmp_leaf_id, :]) // self.TP - np.sum(self.banned_gpu_status[pod_id, tmp_leaf_id, :]) // self.TP, name='x_i')
            used_i[tmp_leaf_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1, name='used_i')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960000, name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()
        # set constraint 1
        model.addConstr(
            gurobipy.quicksum(x_i[tmp_leaf_id] for tmp_leaf_id in range(self.leaf_per_pod)) == used_server_num)
        # set constraint 2
        for tmp_leaf_id in range(self.leaf_per_pod):
            model.addConstr(used_i[tmp_leaf_id] <= x_i[tmp_leaf_id])
            model.addConstr(used_i[tmp_leaf_id] * 10 * self.server_per_leaf >= x_i[tmp_leaf_id])

        model.addConstr(100 * gurobipy.quicksum(used_i[tmp_leaf_id] for tmp_leaf_id in range(self.leaf_per_pod))
                        + gurobipy.quicksum(used_i[tmp_leaf_id] * (
                            self.server_per_leaf - np.sum(self.pod_gpu_status[pod_id, tmp_leaf_id, :]) // self.TP
                            - np.sum(self.banned_gpu_status[pod_id, tmp_leaf_id, :]) // self.TP)
                                for tmp_leaf_id in range(self.leaf_per_pod)) <= obj_val)

        # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            x_i_solution = model.getAttr('X', x_i)
            tmp_pod_gpu_used = np.zeros((self.pod_num, self.leaf_per_pod, self.server_per_leaf * self.TP), dtype=int)
            for tmp_leaf_id in range(self.leaf_per_pod):
                if round(x_i_solution[tmp_leaf_id]) > 0:
                    tmp_flag = 0
                    has_chosen_gpu = 0
                    while has_chosen_gpu < x_i_solution[tmp_leaf_id] * self.TP:
                        if self.pod_gpu_status[pod_id, tmp_leaf_id, tmp_flag] == 1 or self.banned_gpu_status[pod_id,tmp_leaf_id,tmp_flag] == 1:
                            tmp_flag += 1
                        else:
                            self.pod_gpu_status[pod_id, tmp_leaf_id, tmp_flag] = 1
                            tmp_pod_gpu_used[pod_id, tmp_leaf_id, tmp_flag] = 1
                            has_chosen_gpu += 1
                            tmp_flag += 1

            if task_id not in self.task_occupy_gpu_map:
                self.task_occupy_gpu_map[task_id] = np.zeros(
                    (self.pod_num, self.leaf_per_pod, self.server_per_leaf * self.TP), dtype=int)
            self.task_occupy_gpu_map[task_id] += tmp_pod_gpu_used

        else:
            raise RuntimeError("something went wrong in GPU placement: %d, %d, %d, %d" %
                               (pod_id, used_server_num, task_id, np.sum(self.pod_gpu_status[pod_id, :, :])))

    def occupy_resource(self, task_require_server_num, EP_size, task_id):
        model = gurobipy.Model("PodStrategy solution")
        model.setParam('OutputFlag', 0)
        #model.setParam('TimeLimit', 300)
        x_i = {}
        e_i = {}
        used_i = {}

        ep_upper_bound = math.floor(self.gpu_per_pod // EP_size)
        for pod_id in range(self.pod_num):
            x_i[pod_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.server_num_per_pod, name='x_i')
            used_i[pod_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1, name='used_i')
            e_i[pod_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=ep_upper_bound, name='e_i')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960000, name='obj')
        mini_x = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960, name='mini_x')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        # 线性化条件
        for pod_id in range(self.pod_num):
            model.addConstr(x_i[pod_id] <= self.server_num_per_pod - self.busy_server_per_pod[pod_id] - self.banned_server_per_pod[pod_id])
            model.addConstr(x_i[pod_id] >= mini_x)

        model.addConstr(gurobipy.quicksum(x_i[pod_id] for pod_id in range(self.pod_num)) == task_require_server_num)

        for pod_id in range(self.pod_num):
            model.addConstr(used_i[pod_id] <= x_i[pod_id])
            model.addConstr(used_i[pod_id] * 10 * self.server_num_per_pod >= x_i[pod_id])

        model.addConstr(100 * gurobipy.quicksum(used_i[pod_id] for pod_id in range(self.pod_num))
                        + gurobipy.quicksum(used_i[pod_id] * (self.server_num_per_pod - self.busy_server_per_pod[pod_id] - self.banned_server_per_pod[pod_id])
                                            for pod_id in range(self.pod_num)) - mini_x <= obj_val)
        # MOE约束
        for pod_id in range(self.pod_num):
            model.addConstr(e_i[pod_id] * EP_size == x_i[pod_id] * self.TP)

        # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            x_i_solution = model.getAttr('X', x_i)

            res = []
            for i in range(self.pod_num):
                res.append(round(x_i_solution[i]))
                self.busy_server_per_pod[i] += round(x_i_solution[i])
                assert self.busy_server_per_pod[i] <= self.server_num_per_pod
            self.task_occupy_resource_map[task_id] = res
            assert sum(res) == task_require_server_num
            for pod_id in range(self.pod_num):
                used_server_num = round(x_i_solution[pod_id])
                self.occupy_gpu(pod_id, used_server_num, task_id)

            return res, deepcopy(self.task_occupy_gpu_map[task_id])

        else:
            return [], None
        
    def occupy_resource_for_large_EP(self, task_require_server_num, EP_size, task_id):
        model = gurobipy.Model("PodStrategy solution")
        model.setParam('OutputFlag', 0)
        #model.setParam('TimeLimit', 300)
        x_i = {}
        e_i = {}
        used_i = {}
        min_pod_num = math.ceil(EP_size / self.gpu_per_pod)
        
        have_allocated = False
        # 从尽可能少的Pod数量开始遍历
        while not have_allocated and min_pod_num<self.pod_num:
            # 计算当前EP块大小
            cur_ep_size = math.ceil(EP_size / min_pod_num)

            # 计算当前EP块的上限，允许一个Pod放多个EP块
            cur_ep_upper_bound = math.floor(self.gpu_per_pod / cur_ep_size)
        
            for pod_id in range(self.pod_num):
                x_i[pod_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.server_num_per_pod, name='x_i')
                used_i[pod_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1, name='used_i')
                e_i[pod_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=cur_ep_upper_bound, name='e_i')
            obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960000, name='obj')
            mini_x = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960, name='mini_x')
            model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
            model.update()

            # 线性化条件
            for pod_id in range(self.pod_num):
                model.addConstr(x_i[pod_id] <= self.server_num_per_pod - self.busy_server_per_pod[pod_id] - self.banned_server_per_pod[pod_id])
                model.addConstr(x_i[pod_id] >= mini_x)

            model.addConstr(gurobipy.quicksum(x_i[pod_id] for pod_id in range(self.pod_num)) == task_require_server_num)

            for pod_id in range(self.pod_num):
                model.addConstr(used_i[pod_id] <= x_i[pod_id])
                model.addConstr(used_i[pod_id] * 10 * self.server_num_per_pod >= x_i[pod_id])

            model.addConstr(100 * gurobipy.quicksum(used_i[pod_id] for pod_id in range(self.pod_num))
                            + gurobipy.quicksum(used_i[pod_id] * (self.server_num_per_pod - self.busy_server_per_pod[pod_id] - self.banned_server_per_pod[pod_id])
                                                for pod_id in range(self.pod_num)) - mini_x <= obj_val)
            # MOE约束
            for pod_id in range(self.pod_num):
                model.addConstr(e_i[pod_id] * cur_ep_size == x_i[pod_id] * self.TP)

            # 开始执行
            model.update()
            model.optimize()
            # 记录运行结果
            if model.status == gurobipy.GRB.Status.OPTIMAL:
                x_i_solution = model.getAttr('X', x_i)
                
                res = []
                for i in range(self.pod_num):
                    res.append(round(x_i_solution[i]))
                    self.busy_server_per_pod[i] += round(x_i_solution[i])
                    assert self.busy_server_per_pod[i] <= self.server_num_per_pod
                self.task_occupy_resource_map[task_id] = res
                assert sum(res) == task_require_server_num
                for pod_id in range(self.pod_num):
                    used_server_num = round(x_i_solution[pod_id])
                    self.occupy_gpu(pod_id, used_server_num, task_id)
                have_allocated = True
                return res, deepcopy(self.task_occupy_gpu_map[task_id]), cur_ep_size
            else:
                min_pod_num = min_pod_num*2
                
        if not have_allocated:
            return [], None, None

    def release_resource(self, task_id):
        occupy_list = self.task_occupy_resource_map[task_id]
        for i in range(len(occupy_list)):
            self.busy_server_per_pod[i] -= round(occupy_list[i])
            assert self.busy_server_per_pod[i] >= 0
            
        self.pod_gpu_status -= self.task_occupy_gpu_map[task_id]
        
        self.task_occupy_resource_map[task_id] = np.zeros(self.pod_num, dtype=int)
        self.task_occupy_gpu_map[task_id] = np.zeros((self.pod_num, self.leaf_per_pod, self.server_per_leaf * self.TP), dtype=int)

    def random_fail_gpu(self, failure_id):
        # 随机选择正在运行的server并ban掉
        chosen_banned_gpu_list = []
        potentional_gpu_list = []
        gpu_per_leaf = self.server_per_leaf * self.TP
        new_banned_gpu_status = np.zeros((self.pod_num, self.leaf_per_pod, gpu_per_leaf), dtype=int)
        new_banned_server_per_pod = np.zeros(self.pod_num, dtype=int)
        for pod_id in range(self.pod_num):
            for local_leaf_id in range(self.leaf_per_pod):
                for local_gpu_id in range(gpu_per_leaf):
                    if self.pod_gpu_status[pod_id, local_leaf_id, local_gpu_id] == 1 and self.banned_gpu_status[pod_id, local_leaf_id, local_gpu_id] == 0:
                        potentional_gpu_list.append(pod_id * self.leaf_per_pod * gpu_per_leaf + local_leaf_id * gpu_per_leaf + local_gpu_id)
        if len(potentional_gpu_list) == 0:
            return new_banned_gpu_status, -1, new_banned_server_per_pod

        random.seed(failure_id)
        random_banned_gpu = random.choice(potentional_gpu_list)
        chosen_pod_id = random_banned_gpu // self.gpu_per_pod
        chosen_local_leaf_id = (random_banned_gpu % self.gpu_per_pod) // gpu_per_leaf
        chosen_local_server_id = ((random_banned_gpu % self.gpu_per_pod) % gpu_per_leaf) // self.TP
        for potential_gpu_id in potentional_gpu_list:
            cur_pod_id = potential_gpu_id // self.gpu_per_pod
            cur_local_leaf_id = (potential_gpu_id % self.gpu_per_pod) // gpu_per_leaf
            cur_local_server_id = ((potential_gpu_id % self.gpu_per_pod) % gpu_per_leaf ) // self.TP
            cur_local_gpu_id = (potential_gpu_id % self.gpu_per_pod) % gpu_per_leaf
            if cur_pod_id == chosen_pod_id and cur_local_leaf_id == chosen_local_leaf_id and chosen_local_server_id == cur_local_server_id:
                chosen_banned_gpu_list.append((cur_pod_id, cur_local_leaf_id, cur_local_gpu_id))
        assert len(chosen_banned_gpu_list) == self.TP
        # 调度资源时gpu status相加
        for gpu_info_pair in chosen_banned_gpu_list:
            assert self.banned_gpu_status[gpu_info_pair] == 0
            self.banned_gpu_status[gpu_info_pair] = 1
            new_banned_gpu_status[gpu_info_pair] = 1

        # 定位影响的任务
        chosen_task_id = -1
        for task_id, use_gpu_status in self.task_occupy_gpu_map.items():
            if use_gpu_status[chosen_banned_gpu_list[0]] == 1:
                assert chosen_task_id == -1
                chosen_task_id = task_id

        # 更新受影响的pod
        self.banned_server_per_pod[chosen_pod_id] += 1
        new_banned_server_per_pod[chosen_pod_id] += 1
        # print(f'debug add influenced server {np.sum(self.banned_server_per_pod[:])}')

        return new_banned_gpu_status,chosen_task_id,new_banned_server_per_pod

    def repair_fail_gpu(self, new_banned_gpu_status, new_banned_server_per_pod):
        # 调度资源时gpu status相加
        self.banned_gpu_status = self.banned_gpu_status - new_banned_gpu_status
        self.banned_server_per_pod = self.banned_server_per_pod - new_banned_server_per_pod
        # print(f'debug remove influenced server {np.sum(self.banned_server_per_pod[:])}')
        assert np.all(self.banned_gpu_status >= 0)
        assert np.all(self.banned_server_per_pod >= 0)
