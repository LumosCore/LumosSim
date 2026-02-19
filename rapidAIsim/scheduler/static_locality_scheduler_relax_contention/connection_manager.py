# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，这一阶段不涉及connection manager
# 2. 当需要跨leaf通信时，此时有两种情况：

import gurobipy
import numpy as np
import math

class ConnectionManager:
    def __init__(self, gpu_num = 512, server_num = 64, leaf_num = 16, spine_num = 16):
        self.gpu_num = gpu_num
        self.server_num = server_num
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.server_per_leaf = int(server_num/leaf_num)
        self.gpu_per_server = int(gpu_num/server_num)
        self.cur_task_id = 0

        self.leaf_to_spine_remain_port_num = {}
        for leaf_id in range(leaf_num):
            for to_spine_id in range(spine_num):
                if leaf_id not in self.leaf_to_spine_remain_port_num:
                    self.leaf_to_spine_remain_port_num[leaf_id] = {}
                self.leaf_to_spine_remain_port_num[leaf_id][to_spine_id] = 1
        self.leaf_remain_port_map = {}
        for leaf_id in range(leaf_num):
            self.leaf_remain_port_map[leaf_id] = int(self.gpu_num/self.leaf_num)
        self.spine_remain_port_map = {}
        for spine_id in range(spine_num):
            self.spine_remain_port_map[spine_id] = int(self.gpu_num/self.spine_num)

        
        
    # 在情况a中，选择好leaf和spine后就可以更新leaf_to_spine_map
    def find_valid_gpu_for_no_pow2_tas_releax(self, require_gpu_size, leaf_remain_empt_server_list, require_leaf_num, require_spine_num, leaf_remain_empt_gpu_list,link_weight):
        assert require_gpu_size == require_leaf_num*require_spine_num
        model = gurobipy.Model("SpineStrategy solution")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        name_list_x_i = []
        name_list_y_j = []
        name_list_c_i_j = []
        for i in range(self.leaf_num):
            name_list_x_i.append(str(i))
        for j in range(self.spine_num):
            name_list_y_j.append(str(j))
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                    name_list_c_i_j.append(str(i)+'_'+str(j))
        

        # add variable
        x_i = {}
        s_i = {}
        for it in name_list_x_i:
            x_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i')
            s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
        y_j = {}
        for it in name_list_y_j:
            y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='y_j')
        c_i_j = {}
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                c_i_j[str(i)+'_'+str(j)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=max(1,self.leaf_to_spine_remain_port_num[i][j]),name='c_i_j')

        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
        obj_val2 = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
        model.setObjective(obj_val+5*obj_val2, gurobipy.GRB.MINIMIZE)
        model.update()

        # 线性化条件
        model.addConstr(gurobipy.quicksum(x_i[str(i)] for i in range(self.leaf_num)) == require_leaf_num)
        model.addConstr(gurobipy.quicksum(y_j[str(j)] for j in range(self.spine_num)) == require_spine_num)
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                model.addConstr(x_i[str(i)]>=c_i_j[str(i)+'_'+str(j)])
                model.addConstr(y_j[str(j)]>=c_i_j[str(i)+'_'+str(j)])
            model.addConstr(s_i[str(i)]>=x_i[str(i)])
        model.addConstr(gurobipy.quicksum(s_i[str(i)]*self.gpu_per_server for i in range(self.leaf_num)) == require_gpu_size)
        for j in range(self.spine_num):
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for i in range(self.leaf_num)) == require_leaf_num*y_j[str(j)])
        for i in range(self.leaf_num):
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for j in range(self.spine_num)) == require_spine_num*x_i[str(i)])
        for i in range(self.leaf_num):
            model.addConstr(s_i[str(i)]*self.gpu_per_server == int(require_gpu_size/require_leaf_num)*x_i[str(i)])
        for j in range(self.spine_num):
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for i in range(self.leaf_num)) <= self.leaf_num)
        for i in range(self.leaf_num):
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for j in range(self.spine_num)) <= self.spine_num)
                

        model.addConstr(gurobipy.quicksum(leaf_remain_empt_server_list[leaf_id]*x_i[str(leaf_id)] for leaf_id in range(self.leaf_num))<=obj_val)
        model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)]*link_weight[str(i)+'_'+str(j)]*(1-self.leaf_to_spine_remain_port_num[i][j]) for i in range(self.leaf_num) for j in range(self.spine_num) if str(i)+'_'+str(j) in link_weight)<=obj_val2)
        
        print("start running")

        # 开始执行
        model.update()
        model.optimize()
        print("finish running")
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            c_i_j_solution = model.getAttr('X', c_i_j)
            x_i_solution = model.getAttr('X', x_i)
            y_j_solution = model.getAttr('X', y_j)
            s_i_solution = model.getAttr('X', s_i)
            leaf_occupy_gpu_num_map = {}
            leaf_remain_gpu_num_map = {}
            allocation_link_mapping = []
            job_allocated_leaf_spine_link = {}
            for leaf_id in range(self.leaf_num):
                if leaf_id not in job_allocated_leaf_spine_link:
                    job_allocated_leaf_spine_link[leaf_id] = {}
                for spine_id in range(self.spine_num): 
                    if spine_id not in job_allocated_leaf_spine_link[leaf_id]:
                        job_allocated_leaf_spine_link[leaf_id][spine_id] = 0
                    job_allocated_leaf_spine_link[leaf_id][spine_id] += round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>=0:
                    self.leaf_remain_port_map[leaf_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    if round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])>0:
                        if str(leaf_id)+'_'+str(spine_id) not in link_weight:
                            link_weight[str(leaf_id)+'_'+str(spine_id)] = 0
                        link_weight[str(leaf_id)+'_'+str(spine_id)] += require_gpu_size
            for leaf_id in range(self.leaf_num):
                for spine_id in range(self.spine_num):
                    if c_i_j_solution[str(leaf_id)+'_'+str(spine_id)]>0:
                        # assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>0
                        self.leaf_to_spine_remain_port_num[leaf_id][spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                        allocation_link_mapping.append([self.gpu_num+leaf_id, self.gpu_num+self.leaf_num+spine_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
                        allocation_link_mapping.append([self.gpu_num+self.leaf_num+spine_id, self.gpu_num+leaf_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
            for leaf_id in range(self.leaf_num):
                if round(x_i_solution[str(leaf_id)]):
                    leaf_occupy_gpu_num_map[leaf_id] = round(s_i_solution[str(leaf_id)]*self.gpu_per_server)
            contention_link = {}
            link_contention_res = 0
            for leaf_id in range(self.leaf_num):
                for spine_id in range(self.spine_num):
                    if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]<0:
                        link_contention_res += 1
                        if f'{leaf_id}_{spine_id}' not in contention_link:
                            contention_link[f'{leaf_id}_{spine_id}'] = 0
                        contention_link[f'{leaf_id}_{spine_id}'] += 1
            if link_contention_res>0:
                print("return contention res:",link_contention_res)
                print(allocation_link_mapping)
            self.cur_task_id += 1
            return True, allocation_link_mapping, leaf_occupy_gpu_num_map, job_allocated_leaf_spine_link,contention_link
        else:
            assert False
            print("fuck0", require_leaf_num, require_spine_num)
            print(leaf_remain_empt_gpu_list)
            print(self.leaf_remain_port_map)
            print(leaf_remain_empt_server_list)
            require_gpu_size, leaf_remain_empt_server_list, require_leaf_num, require_spine_num, leaf_remain_empt_gpu_list
            return False, None, None, None, None
        
    def find_valid_gpu_for_no_pow2_tas_releax_iter(self, require_gpu_size, leaf_remain_empt_server_list, start_require_leaf_num,leaf_remain_empt_gpu_list,link_weight):
        cur_loss = 10000000
        best_c_i_j_solution = None
        best_x_i_solution = None
        best_y_j_solution = None
        best_s_i_solution = None
        require_leaf_num = start_require_leaf_num
        while require_leaf_num < require_gpu_size and require_leaf_num <= self.leaf_num:
            require_spine_num = int(require_gpu_size/require_leaf_num)
            potentional_leaf_list = []
            # Step1. 在leaf_resource_manager中选取合适的leafgroup
            for temp_leaf_id in range(self.leaf_num):
                require_server_num = math.ceil(require_spine_num/self.gpu_per_server)
                if leaf_remain_empt_server_list[temp_leaf_id]>=require_server_num:
                    potentional_leaf_list.append([temp_leaf_id])
            if len(potentional_leaf_list) < require_leaf_num:
                require_leaf_num*=2
            else:
                model = gurobipy.Model("SpineStrategy solution")
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', 300)
                name_list_x_i = []
                name_list_y_j = []
                name_list_c_i_j = []
                for i in range(self.leaf_num):
                    name_list_x_i.append(str(i))
                for j in range(self.spine_num):
                    name_list_y_j.append(str(j))
                for i in range(self.leaf_num):
                    for j in range(self.spine_num):
                            name_list_c_i_j.append(str(i)+'_'+str(j))
                

                # add variable
                x_i = {}
                s_i = {}
                for it in name_list_x_i:
                    x_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i')
                    s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
                y_j = {}
                for it in name_list_y_j:
                    y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='y_j')
                c_i_j = {}
                for i in range(self.leaf_num):
                    for j in range(self.spine_num):
                        c_i_j[str(i)+'_'+str(j)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=max(1,self.leaf_to_spine_remain_port_num[i][j]),name='c_i_j')

                obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
                obj_val2 = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
                model.setObjective((6+0.3*(1/(1+pow(2.71828,(require_gpu_size-self.gpu_per_server*self.server_per_leaf)))))*obj_val+1*obj_val2, gurobipy.GRB.MINIMIZE)
                model.update()

                # 线性化条件
                model.addConstr(gurobipy.quicksum(x_i[str(i)] for i in range(self.leaf_num)) == require_leaf_num)
                model.addConstr(gurobipy.quicksum(y_j[str(j)] for j in range(self.spine_num)) == require_spine_num)
                for i in range(self.leaf_num):
                    for j in range(self.spine_num):
                        model.addConstr(x_i[str(i)]>=c_i_j[str(i)+'_'+str(j)])
                        model.addConstr(y_j[str(j)]>=c_i_j[str(i)+'_'+str(j)])
                    model.addConstr(s_i[str(i)]>=x_i[str(i)])
                model.addConstr(gurobipy.quicksum(s_i[str(i)]*self.gpu_per_server for i in range(self.leaf_num)) == require_gpu_size)
                for j in range(self.spine_num):
                    model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for i in range(self.leaf_num)) == require_leaf_num*y_j[str(j)])
                for i in range(self.leaf_num):
                    model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for j in range(self.spine_num)) == require_spine_num*x_i[str(i)])
                for i in range(self.leaf_num):
                    model.addConstr(s_i[str(i)]*self.gpu_per_server == int(require_gpu_size/require_leaf_num)*x_i[str(i)])
                for j in range(self.spine_num):
                    model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for i in range(self.leaf_num)) <= self.leaf_num)
                for i in range(self.leaf_num):
                    model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for j in range(self.spine_num)) <= self.spine_num)
                        

                model.addConstr(gurobipy.quicksum(leaf_remain_empt_server_list[leaf_id]*x_i[str(leaf_id)] for leaf_id in range(self.leaf_num))
                                + (0.004+0.002*self.cur_task_id/5000)*gurobipy.quicksum(max(0,self.spine_remain_port_map[spine_id])*y_j[str(spine_id)] for spine_id in range(self.spine_num))<=obj_val)
                model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)]*link_weight[str(i)+'_'+str(j)] for i in range(self.leaf_num) for j in range(self.spine_num) if str(i)+'_'+str(j) in link_weight)<=obj_val2)
                
                print("start running",require_leaf_num, require_gpu_size)

                # 开始执行
                model.update()
                model.optimize()
                print("finish running")
                require_leaf_num *=2
                # 记录运行结果
                if model.status == gurobipy.GRB.Status.OPTIMAL:
                    obj_val = int(model.ObjVal)/(1+require_leaf_num/self.leaf_num)
                    if obj_val < cur_loss:
                        cur_loss = obj_val
                        best_c_i_j_solution = model.getAttr('X', c_i_j)
                        best_x_i_solution = model.getAttr('X', x_i)
                        best_y_j_solution = model.getAttr('X', y_j)
                        best_s_i_solution = model.getAttr('X', s_i)

        if cur_loss == 10000000 or cur_loss>512:
            return False, None, None, None, None
        self.cur_task_id += 1
        print("debug obj", cur_loss,self.cur_task_id)
        c_i_j_solution = best_c_i_j_solution
        x_i_solution = best_x_i_solution
        y_j_solution = best_y_j_solution
        s_i_solution = best_s_i_solution
        leaf_occupy_gpu_num_map = {}
        leaf_remain_gpu_num_map = {}
        allocation_link_mapping = []
        job_allocated_leaf_spine_link = {}
        for leaf_id in range(self.leaf_num):
            if leaf_id not in job_allocated_leaf_spine_link:
                job_allocated_leaf_spine_link[leaf_id] = {}
            for spine_id in range(self.spine_num): 
                if spine_id not in job_allocated_leaf_spine_link[leaf_id]:
                    job_allocated_leaf_spine_link[leaf_id][spine_id] = 0
                job_allocated_leaf_spine_link[leaf_id][spine_id] += round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>=0:
                self.leaf_remain_port_map[leaf_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                self.spine_remain_port_map[spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                if round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])>0:
                    if str(leaf_id)+'_'+str(spine_id) not in link_weight:
                        link_weight[str(leaf_id)+'_'+str(spine_id)] = 0
                    link_weight[str(leaf_id)+'_'+str(spine_id)] += require_gpu_size
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                if c_i_j_solution[str(leaf_id)+'_'+str(spine_id)]>0:
                    # assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>0
                    self.leaf_to_spine_remain_port_num[leaf_id][spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    allocation_link_mapping.append([self.gpu_num+leaf_id, self.gpu_num+self.leaf_num+spine_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
                    allocation_link_mapping.append([self.gpu_num+self.leaf_num+spine_id, self.gpu_num+leaf_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
        for leaf_id in range(self.leaf_num):
            if round(x_i_solution[str(leaf_id)]):
                leaf_occupy_gpu_num_map[leaf_id] = round(s_i_solution[str(leaf_id)]*self.gpu_per_server)
        contention_link = {}
        link_contention_res = 0
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]<0:
                    link_contention_res += 1
                    if f'{leaf_id}_{spine_id}' not in contention_link:
                        contention_link[f'{leaf_id}_{spine_id}'] = 0
                    contention_link[f'{leaf_id}_{spine_id}'] += 1
        if link_contention_res>0:
            print("return contention res:",link_contention_res)
            print(allocation_link_mapping)
        return True, allocation_link_mapping, leaf_occupy_gpu_num_map, job_allocated_leaf_spine_link,contention_link        
        

    def print_connection_info(self):
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id] >0:
                print(leaf_id,spine_id,self.leaf_to_spine_remain_port_num[leaf_id][spine_id])

    def release_connection_resource(self, job_allocated_leaf_spine_link):
        for leaf_id in job_allocated_leaf_spine_link:
            for spine_id in job_allocated_leaf_spine_link[leaf_id]:
                self.leaf_to_spine_remain_port_num[leaf_id][spine_id] += job_allocated_leaf_spine_link[leaf_id][spine_id]
                self.leaf_remain_port_map[leaf_id] += job_allocated_leaf_spine_link[leaf_id][spine_id]
                self.spine_remain_port_map[spine_id] += job_allocated_leaf_spine_link[leaf_id][spine_id]


# connection_manager_ = ConnectionManager(512, 64, 32, 15)
# connection_manager_.find_valid_gpu_for_no_pow2_tas_releax()