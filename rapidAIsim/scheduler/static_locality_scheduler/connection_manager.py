# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，这一阶段不涉及connection manager
# 2. 当需要跨leaf通信时，此时有两种情况：

import gurobipy
import numpy as np

class ConnectionManager:
    def __init__(self, gpu_num = 512, server_num = 64, leaf_num = 16, spine_num = 16):
        self.gpu_num = gpu_num
        self.server_num = server_num
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.server_per_leaf = int(server_num/leaf_num)
        self.gpu_per_server = int(gpu_num/server_num)

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

    def update_leaf_to_spine_map_according_to_given_clos(self, chosen_leaf_id_num_list, choosed_spine_index_list, gpu_num,job_allocated_leaf_spine_link):
        temp_link_num = 0
        allocation_link_mapping = []
        for leaf_id_num_pair in chosen_leaf_id_num_list:
            leaf_id = leaf_id_num_pair[0]
            for spine_id in choosed_spine_index_list:
                need_chosen_link = int(gpu_num/len(choosed_spine_index_list))
                assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id] >= need_chosen_link
                self.leaf_to_spine_remain_port_num[leaf_id][spine_id] -= need_chosen_link
                if leaf_id not in job_allocated_leaf_spine_link:
                    job_allocated_leaf_spine_link[leaf_id] = {}
                if spine_id not in job_allocated_leaf_spine_link[leaf_id]:
                    job_allocated_leaf_spine_link[leaf_id][spine_id] = 0
                job_allocated_leaf_spine_link[leaf_id][spine_id] += need_chosen_link
                temp_link_num += need_chosen_link
                allocation_link_mapping.append([self.gpu_num+leaf_id, self.gpu_num+self.leaf_num+spine_id, need_chosen_link])
                allocation_link_mapping.append([self.gpu_num+self.leaf_num+spine_id, self.gpu_num+leaf_id, need_chosen_link])
        assert temp_link_num == gpu_num
        return allocation_link_mapping

    # 在情况a中，选择好leaf和spine后就可以更新leaf_to_spine_map
    def update_leaf_to_spine_map_according_to_gpu_size(self, require_gpu_size, leaf_remain_empt_server_list, require_leaf_num, require_spine_num):
        self.check_valid_network()
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
                c_i_j[str(i)+'_'+str(j)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.leaf_to_spine_remain_port_num[i][j],name='c_i_j')

        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
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
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for i in range(self.leaf_num)) <= self.spine_remain_port_map[j])
        for i in range(self.leaf_num):
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for j in range(self.spine_num)) <= self.leaf_remain_port_map[i])

        model.addConstr(gurobipy.quicksum(self.spine_remain_port_map[spine_id]*y_j[str(spine_id)] for spine_id in range(self.spine_num))
                        +gurobipy.quicksum(self.leaf_remain_port_map[leaf_id]*x_i[str(leaf_id)] for leaf_id in range(self.leaf_num))<=obj_val)

        
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
            spine_occupy_port_num_map = {}
            allocation_link_mapping = []
            for leaf_id in range(self.leaf_num):
                for spine_id in range(self.spine_num):
                    if c_i_j_solution[str(leaf_id)+'_'+str(spine_id)]>0:
                        assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>0
                        self.leaf_to_spine_remain_port_num[leaf_id][spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                        allocation_link_mapping.append([self.gpu_num+leaf_id, self.gpu_num+self.leaf_num+spine_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
                        allocation_link_mapping.append([self.gpu_num+self.leaf_num+spine_id, self.gpu_num+leaf_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
            for leaf_id in range(self.leaf_num):
                if x_i_solution[str(leaf_id)]:
                    leaf_occupy_gpu_num_map[leaf_id] = round(s_i_solution[str(leaf_id)]*self.gpu_per_server)
            for spine_id in range(self.spine_num):
                if y_j_solution[str(spine_id)]>0:
                    spine_occupy_port_num_map[spine_id] = round(require_leaf_num*y_j_solution[str(spine_id)])
            job_allocated_leaf_spine_link = {}
            for leaf_id in range(self.leaf_num):
                if leaf_id not in job_allocated_leaf_spine_link:
                    job_allocated_leaf_spine_link[leaf_id] = {}
                for spine_id in range(self.spine_num): 
                    if spine_id not in job_allocated_leaf_spine_link[leaf_id]:
                        job_allocated_leaf_spine_link[leaf_id][spine_id] = 0
                    job_allocated_leaf_spine_link[leaf_id][spine_id] += round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    self.leaf_remain_port_map[leaf_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    self.spine_remain_port_map[spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
            print("return result")
            return True, allocation_link_mapping, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, job_allocated_leaf_spine_link
        else:
            # print(self.leaf_to_spine_remain_port_num)
            # exit()
            print("fuck0", require_leaf_num, require_spine_num)
            print(self.spine_remain_port_map)
            print(self.leaf_remain_port_map)
            print(leaf_remain_empt_server_list)
            #self.print_connection_info()
            return False, None, None, None, None
        
    # 在情况a中，选择好leaf和spine后就可以更新leaf_to_spine_map
    def find_valid_gpu_for_no_pow2_task(self, require_gpu_size, leaf_remain_empt_server_list, require_leaf_num, require_spine_num, leaf_remain_empt_gpu_list):
        self.check_valid_network()
        remain_chosen_gpu_num = 0
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
        x_r_i = {}
        temp_x_r_i = {}
        for it in name_list_x_i:
            x_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i')
            s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
            x_r_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_gpu_list[int(it)],name='x_r_i')
            temp_x_r_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=int(leaf_remain_empt_gpu_list[int(it)]/2),name='temp_x_r_i')
        y_j = {}
        for it in name_list_y_j:
            y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='y_j')
        c_i_j = {}
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                c_i_j[str(i)+'_'+str(j)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.leaf_to_spine_remain_port_num[i][j],name='c_i_j')

        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
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
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for i in range(self.leaf_num)) <= self.spine_remain_port_map[j])
        for i in range(self.leaf_num):
            model.addConstr(gurobipy.quicksum(c_i_j[str(i)+'_'+str(j)] for j in range(self.spine_num)) <= self.leaf_remain_port_map[i])
        model.addConstr( gurobipy.quicksum( x_r_i[str(i)] for i in range(self.leaf_num) ) == remain_chosen_gpu_num )
        for i in range(self.leaf_num):
            model.addConstr(  x_r_i[str(i)] == 2*temp_x_r_i[str(i)] )
            model.addConstr(  x_r_i[str(i)] + x_i[str(i)]*require_spine_num <= leaf_remain_empt_gpu_list[i] )
            model.addConstr(  x_r_i[str(i)] <= x_i[str(i)]*require_spine_num )
                

        # model.addConstr(gurobipy.quicksum(self.spine_remain_port_map[spine_id]*y_j[str(spine_id)] for spine_id in range(self.spine_num))
        #                 +gurobipy.quicksum(self.leaf_remain_port_map[leaf_id]*x_i[str(leaf_id)] for leaf_id in range(self.leaf_num))<=obj_val)
        model.addConstr(gurobipy.quicksum(self.leaf_remain_port_map[leaf_id]*x_i[str(leaf_id)] for leaf_id in range(self.leaf_num))<=obj_val)

        
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
            x_r_i_solution = model.getAttr('X', x_r_i)
            leaf_occupy_gpu_num_map = {}
            leaf_remain_gpu_num_map = {}
            spine_occupy_port_num_map = {}
            allocation_link_mapping = []
            for leaf_id in range(self.leaf_num):
                if round(x_r_i_solution[str(leaf_id)])>0:
                    if i not in leaf_remain_gpu_num_map:
                        leaf_remain_gpu_num_map[leaf_id] = 0
                    leaf_remain_gpu_num_map[leaf_id] += round(x_r_i_solution[str(leaf_id)])
                for spine_id in range(self.spine_num):
                    if c_i_j_solution[str(leaf_id)+'_'+str(spine_id)]>0:
                        assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id]>0
                        self.leaf_to_spine_remain_port_num[leaf_id][spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                        allocation_link_mapping.append([self.gpu_num+leaf_id, self.gpu_num+self.leaf_num+spine_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
                        allocation_link_mapping.append([self.gpu_num+self.leaf_num+spine_id, self.gpu_num+leaf_id, round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])])
            for leaf_id in range(self.leaf_num):
                if round(x_i_solution[str(leaf_id)]):
                    leaf_occupy_gpu_num_map[leaf_id] = round(s_i_solution[str(leaf_id)]*self.gpu_per_server)
            for spine_id in range(self.spine_num):
                if round(y_j_solution[str(spine_id)])>0:
                    spine_occupy_port_num_map[spine_id] = round(require_leaf_num*y_j_solution[str(spine_id)])
            job_allocated_leaf_spine_link = {}
            for leaf_id in range(self.leaf_num):
                if leaf_id not in job_allocated_leaf_spine_link:
                    job_allocated_leaf_spine_link[leaf_id] = {}
                for spine_id in range(self.spine_num): 
                    if spine_id not in job_allocated_leaf_spine_link[leaf_id]:
                        job_allocated_leaf_spine_link[leaf_id][spine_id] = 0
                    job_allocated_leaf_spine_link[leaf_id][spine_id] += round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    self.leaf_remain_port_map[leaf_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
                    self.spine_remain_port_map[spine_id] -= round(c_i_j_solution[str(leaf_id)+'_'+str(spine_id)])
            print("return result")
            return True, allocation_link_mapping, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, job_allocated_leaf_spine_link, leaf_remain_gpu_num_map
        else:
            # print(self.leaf_to_spine_remain_port_num)
            # exit()
            print("fuck0", require_leaf_num, require_spine_num)
            print(leaf_remain_empt_gpu_list)
            print(self.spine_remain_port_map)
            print(self.leaf_remain_port_map)
            print(leaf_remain_empt_server_list)
            #self.print_connection_info()
            return False, None, None, None, None, None

    def print_connection_info(self):
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                # if self.leaf_to_spine_remain_port_num[leaf_id][spine_id] >0:
                print(leaf_id,spine_id,self.leaf_to_spine_remain_port_num[leaf_id][spine_id])

    def release_connection_resource(self, job_allocated_leaf_spine_link):
        self.check_valid_network()
        for leaf_id in job_allocated_leaf_spine_link:
            for spine_id in job_allocated_leaf_spine_link[leaf_id]:
                self.leaf_to_spine_remain_port_num[leaf_id][spine_id] += job_allocated_leaf_spine_link[leaf_id][spine_id]
                self.leaf_remain_port_map[leaf_id] += job_allocated_leaf_spine_link[leaf_id][spine_id]
                self.spine_remain_port_map[spine_id] += job_allocated_leaf_spine_link[leaf_id][spine_id]
                if self.leaf_to_spine_remain_port_num[leaf_id][spine_id]<0:
                    print(job_allocated_leaf_spine_link[leaf_id][spine_id])
                assert self.leaf_to_spine_remain_port_num[leaf_id][spine_id] >= 0
        self.check_valid_network()

    def check_valid_network(self):
        temp_spine_port_remain = {}
        for spine_id in range(self.spine_num):
            if spine_id not in temp_spine_port_remain:
                temp_spine_port_remain[spine_id] = 0 
        for spine_id in range(self.spine_num):
            for leaf_id in range(self.leaf_num):
                temp_spine_port_remain[spine_id] += self.leaf_to_spine_remain_port_num[leaf_id][spine_id]
        for spine_id in range(self.spine_num):
            if temp_spine_port_remain[spine_id] != self.spine_remain_port_map[spine_id]:
                print(self.leaf_num, self.spine_num)
                print(spine_id, temp_spine_port_remain[spine_id] ,self.spine_remain_port_map[spine_id])
            assert temp_spine_port_remain[spine_id] == self.spine_remain_port_map[spine_id]