# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，这一阶段不涉及connection manager
# 2. 当需要跨leaf通信时，此时有两种情况：
#   a. 若某几个跨leaf的gpu可以连到同一个spine下，那么当确定了leaf到spine的连接
#   关系后，调用整数规划求出oxc配置方案, 这一过程先改动leaf_to_spine_map
#   b. 若需要spine迁移，从spine manager处得到迁移方案，传到gpu_placementer中，
#   gpu_placementer根据记录的job信息决定spine端口的迁移方案，即更新leaf_to_spine_map，
#   更新任务记录的相关信息，然后调用整数规划求出oxc配置方案。这一过程第一步做spine迁移时
#   不需要改动self.oxc_leaf_spine_map ，但会改动任务占用的线路，以及leaf_to_spine_map，然后
#   才更新连接关系
# 这两种情况都是通过更新（或者不更新）oxc_leaf_spine_map以及新的leaf_to_spine_map确定
# 新的oxc_leaf_spine_map
from operator import mod
import gurobipy
from math import ceil
import numpy as np
from ortools.linear_solver import pywraplp


class ConnectionManager:
    def __init__(self, gpu_num = 512, server_num = 64, leaf_num = 16, spine_num = 8, oxc_num = 32):
        self.gpu_num = gpu_num
        self.server_num = server_num
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.oxc_num = oxc_num
        self.server_per_leaf = int(server_num/leaf_num)
        self.gpu_per_server = int(gpu_num/server_num)
        self.gpu_per_leaf = int(gpu_num/leaf_num)
        self.port_per_spine = int(gpu_num/spine_num)

        # self.server_resource_manager_ = server_resource_manager.ServerResourceManager(server_num, self.gpu_per_server)
        # self.leaf_resource_manager_ = leaf_resource_manager.LeafResourceManager(leaf_num, self.gpu_per_leaf)
        # self.spine_resource_manager_ = spine_resource_manager.SpineSwitchManager(spine_num, self.port_per_spine)

        self.oxc_leaf_spine_map = {}
        for oxc_id in range(oxc_num):
            if oxc_id not in self.oxc_leaf_spine_map:
                self.oxc_leaf_spine_map[oxc_id] = {}
            for leaf_id in range(leaf_num):
                self.oxc_leaf_spine_map[oxc_id][leaf_id] = -1
        self.leaf_to_spine_map = {}
        for leaf_id in range(leaf_num):
            for to_spine_id in range(spine_num):
                if leaf_id not in self.leaf_to_spine_map:
                    self.leaf_to_spine_map[leaf_id] = {}
                self.leaf_to_spine_map[leaf_id][to_spine_id] = 0
                
    def clear_spine_and_oxc(self):
        self.oxc_leaf_spine_map = {}
        for oxc_id in range(self.oxc_num):
            if oxc_id not in self.oxc_leaf_spine_map:
                self.oxc_leaf_spine_map[oxc_id] = {}
            for leaf_id in range(self.leaf_num):
                self.oxc_leaf_spine_map[oxc_id][leaf_id] = -1
        self.leaf_to_spine_map = {}
        for leaf_id in range(self.leaf_num):
            for to_spine_id in range(self.spine_num):
                if leaf_id not in self.leaf_to_spine_map:
                    self.leaf_to_spine_map[leaf_id] = {}
                self.leaf_to_spine_map[leaf_id][to_spine_id] = 0

    # 在情况a中，选择好leaf和spine后就可以更新leaf_to_spine_map，然后调用整数规划配置oxc
    def update_leaf_to_spine_map_according_to_chosen_leaf_and_spine(self, chosen_leaf_id_num_list, choosed_spine_index_list, sim_time=-1):
        # 根据选择的leaf交换机和spine交换机，可以确定该任务的clos形状
        temp_leaf_to_spine_map = {} # key 为leaf的index，value为另一个map B， map B的key为spine交换机的index，value为该leaf要新连多少根线到该spine
        for choosed_leaf_id_num_pair in chosen_leaf_id_num_list:
            temp_leaf_to_each_spine_map = {}
            for choosed_spine_index in choosed_spine_index_list:
                temp_leaf_to_each_spine_map[choosed_spine_index] = int(choosed_leaf_id_num_pair[1]/len(choosed_spine_index_list))
            temp_leaf_to_spine_map[choosed_leaf_id_num_pair[0]] = temp_leaf_to_each_spine_map
        # 首先根据选择的leaf和spine交换机，更新leaf_to_spine_map
        for leaf_switch_index in temp_leaf_to_spine_map:
            for spine_switch_index in temp_leaf_to_spine_map[leaf_switch_index]:
                self.leaf_to_spine_map[leaf_switch_index][spine_switch_index] += temp_leaf_to_spine_map[leaf_switch_index][spine_switch_index]
        # 然后调用整数规划更新oxc_down_to_up_map
        self.oxc_leaf_spine_map, job_allocated_oxc_spine_link  = self.update_oxc_leaf_spine_map(sim_time)
        return self.oxc_leaf_spine_map , temp_leaf_to_spine_map, job_allocated_oxc_spine_link

    # 在情况b中，gpu placementer会根据spine migration方案更新leaf_to_spine_map
    def update_connection_according_to_migration(self, leaf_id, spine_id, change_num, oxc_id):
        self.leaf_to_spine_map[leaf_id][spine_id] += change_num
        assert self.leaf_to_spine_map[leaf_id][spine_id] >= 0
        assert self.leaf_to_spine_map[leaf_id][spine_id] <= int(self.gpu_num/self.leaf_num)

    # 在情况b中，gpu placementer会根据新任务的形状更新leaf_to_spine_map
    def update_leaf_to_spine_map_according_to_new_job(self, temp_leaf_to_spine_map):
        for leaf_switch_index in temp_leaf_to_spine_map:
            for spine_switch_index in temp_leaf_to_spine_map[leaf_switch_index]:
                self.leaf_to_spine_map[leaf_switch_index][spine_switch_index] += temp_leaf_to_spine_map[leaf_switch_index][spine_switch_index]

    def release_connection_resource(self, job_oxc_leaf_spine_map):
        job_leaf_to_spine_map = {}
        for oxc_id in job_oxc_leaf_spine_map:
            for leaf_id in job_oxc_leaf_spine_map[oxc_id]:
                if leaf_id not in job_leaf_to_spine_map:
                    job_leaf_to_spine_map[leaf_id] = {}
                spine_id = job_oxc_leaf_spine_map[oxc_id][leaf_id]
                if spine_id not in job_leaf_to_spine_map[leaf_id]:
                    job_leaf_to_spine_map[leaf_id][spine_id] = 0
                job_leaf_to_spine_map[leaf_id][spine_id] += 1
        for leaf_id in job_leaf_to_spine_map:
            for spine_id in job_leaf_to_spine_map[leaf_id]:
                self.leaf_to_spine_map[leaf_id][spine_id] -= job_leaf_to_spine_map[leaf_id][spine_id]
                assert self.leaf_to_spine_map[leaf_id][spine_id]>=0
        for oxc_id in job_oxc_leaf_spine_map:
            for leaf_id in job_oxc_leaf_spine_map[oxc_id]:
                self.oxc_leaf_spine_map[oxc_id][leaf_id] = -1

    def print_leaf_to_spine_map(self):
        for leaf_id in self.leaf_to_spine_map:
            print(leaf_id,  self.leaf_to_spine_map[leaf_id])
        leaf_num_map = {}
        spine_num_map = {}
        for leaf_id in self.leaf_to_spine_map:
            for spine_id in  self.leaf_to_spine_map[leaf_id]:
                if leaf_id not in leaf_num_map:
                    leaf_num_map[leaf_id] = 0
                leaf_num_map[leaf_id]+=self.leaf_to_spine_map[leaf_id][spine_id]
                if spine_id not in spine_num_map:
                    spine_num_map[spine_id] = 0
                spine_num_map[spine_id]+=self.leaf_to_spine_map[leaf_id][spine_id]
        print(leaf_num_map)
        print(spine_num_map)
        

    def check_leaf_to_spine_map(self):
        for leaf_id in self.leaf_to_spine_map:
            print(leaf_id,  self.leaf_to_spine_map[leaf_id])
        leaf_num_map = {}
        spine_num_map = {}
        for leaf_id in self.leaf_to_spine_map:
            for spine_id in  self.leaf_to_spine_map[leaf_id]:
                if leaf_id not in leaf_num_map:
                    leaf_num_map[leaf_id] = 0
                leaf_num_map[leaf_id]+=self.leaf_to_spine_map[leaf_id][spine_id]
                if spine_id not in spine_num_map:
                    spine_num_map[spine_id] = 0
                spine_num_map[spine_id]+=self.leaf_to_spine_map[leaf_id][spine_id]
        print(leaf_num_map)
        print(spine_num_map)
        for item in leaf_num_map:
            assert leaf_num_map[item]<=self.gpu_per_leaf
        for item in spine_num_map:
            assert spine_num_map[item]<=self.port_per_spine
            
    def find_valid_gpu_for_specific_spine(self, require_gpu_num, require_spine_id, server_remain_gpuNum_map,job_allocated_oxc_spine_link, used_spine_port_num_pair, leaf_remain_empt_server_list):
        oxc_whether_valid = [2 for i in range(self.oxc_num)]
        Z_leafId_oxcId = {}
        for oxc_id in self.oxc_leaf_spine_map:
            for leaf_id in self.oxc_leaf_spine_map[oxc_id]:
                spine_id = self.oxc_leaf_spine_map[oxc_id][leaf_id]
                if spine_id == -1:
                    Z_leafId_oxcId[(leaf_id,oxc_id)] = 0
                else:
                    Z_leafId_oxcId[(leaf_id,oxc_id)] = 1
                if spine_id == require_spine_id:
                    oxc_whether_valid[oxc_id] -= 1
                assert oxc_whether_valid[oxc_id]>=0
        # print(require_spine_id,end=": ")
        # print()
        # for oxc_id_index, value in  enumerate(oxc_whether_valid):
        #     if value>0:
        #         print((oxc_id_index,value),end=", ")
        #         print(self.oxc_leaf_spine_map[oxc_id_index])
        # print()
        require_gpu_within_server = ceil(require_gpu_num/self.server_num)
        
        m = gurobipy.Model("SpineStrategy solution")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 300)
        name_list_x_i = []
        name_list_y_j = []
        name_list_z_i_j = []
        for i in range(self.server_num):
            name_list_x_i.append(str(i))
        for j in range(self.oxc_num):
            name_list_y_j.append(str(j))
        for i in range(self.server_num):
            for j in range(self.oxc_num):
                    name_list_z_i_j.append(str(i)+'_'+str(j))
        x_i = {}
        for it in name_list_x_i:
            x_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i')
        y_j = {}
        for it in name_list_y_j:
            y_j[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=2,name='y_j')
        z_i_j = {}
        for it in name_list_z_i_j:
            z_i_j[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='z_i_j')
        xnum_i = {}
        for it in name_list_x_i:
            xnum_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='xnum_i')
        # z = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1000,name='obj1')
        obj_val = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
        m.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        m.update()

        # 线性化条件
        for j in range(self.oxc_num):
            m.addConstr(y_j[str(j)] <= oxc_whether_valid[j])

        for j in range(self.oxc_num):
            m.addConstr(gurobipy.quicksum(z_i_j[str(i)+'_'+str(j)] for i in range(self.server_num))  <= y_j[str(j)])

        for i in range(self.server_num):
            for j in range(self.oxc_num):
                m.addConstr(z_i_j[str(i)+'_'+str(j)]+ Z_leafId_oxcId[(int(i/self.server_per_leaf),j)]  + gurobipy.quicksum(z_i_j[str(k)+'_'+str(j)] for k in range(self.server_num) if int(i/self.server_per_leaf) == int(k/self.server_per_leaf) and k!=i )<=1)

        for i in range(self.server_num):
            m.addConstr(gurobipy.quicksum( z_i_j[str(i)+'_'+str(j)] for j in range(self.oxc_num)) <= server_remain_gpuNum_map[i])

        m.addConstr(gurobipy.quicksum(z_i_j[str(i)+'_'+str(j)] for i in range(self.server_num) for j in range(self.oxc_num)) == require_gpu_num)

        m.addConstr(gurobipy.quicksum(y_j[str(j)] for j in range(self.oxc_num)) == require_gpu_num)

        for i in range(self.server_num):
            m.addConstr(require_gpu_within_server*xnum_i[str(i)]>=x_i[str(i)])
        
        for i in range(self.server_num):
            m.addConstr(require_gpu_within_server*xnum_i[str(i)]== gurobipy.quicksum( z_i_j[str(i)+'_'+str(j)] for j in range(self.oxc_num)) )

        for i in range(self.server_num):
            for j in range(self.oxc_num):
                m.addConstr(z_i_j[str(i)+'_'+str(j)] <= x_i[str(i)])
                m.addConstr(z_i_j[str(i)+'_'+str(j)] <= y_j[str(j)])
                
        m.addConstr(gurobipy.quicksum(x_i[str(i)]*require_gpu_within_server for i in range(self.server_num)) == require_gpu_num)
        # 目标函数
        leaf_num_k = {}
        for k in range(self.leaf_num):
            leaf_num_k[str(k)] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='leaf_num'+str(k))
        for i in range(self.server_num):
            m.addConstr(leaf_num_k[str(int(i/self.server_per_leaf))]>=x_i[str(i)])
        m.addConstr(obj_val >= gurobipy.quicksum( leaf_num_k[str(k)] for k in range(self.leaf_num)))

        # 开始执行
        m.update()
        m.optimize()
        # 记录运行结果
        if m.status == gurobipy.GRB.Status.OPTIMAL:
            Z_i_j_solution = m.getAttr('X',z_i_j)
            xnum_i_solution = m.getAttr('X',xnum_i)
            #print("fuckfuck", gurobipy.quicksum(x_i[str(i)] for i in range(self.server_num)).getValue(), require_gpu_num)
            assert(round(gurobipy.quicksum(x_i[str(i)] for i in range(self.server_num)).getValue())*require_gpu_within_server == require_gpu_num)
            # 根据Z_i_j_solution更新self.oxc_leaf_spine_map
            for it in name_list_z_i_j:
                divid_index = it.split("_")
                for id in range(len(divid_index)):
                    divid_index[id] = int(divid_index[id])
                if round(Z_i_j_solution[it]) == 1:
                    chosen_server_id = divid_index[0]
                    chosen_oxc_id = divid_index[1]
                    chosen_leaf_id = int(chosen_server_id/self.server_per_leaf)
                    if chosen_oxc_id not in job_allocated_oxc_spine_link:
                        job_allocated_oxc_spine_link[chosen_oxc_id] = {}
                    if chosen_leaf_id not in job_allocated_oxc_spine_link[chosen_oxc_id]:
                        job_allocated_oxc_spine_link[chosen_oxc_id][chosen_leaf_id] = {}
                    job_allocated_oxc_spine_link[chosen_oxc_id][chosen_leaf_id] = require_spine_id
                    self.leaf_to_spine_map[chosen_leaf_id][require_spine_id] += 1
                    assert self.oxc_leaf_spine_map [chosen_oxc_id][chosen_leaf_id] == -1
                    self.oxc_leaf_spine_map [chosen_oxc_id][chosen_leaf_id] = require_spine_id
                used_spine_port_num_pair[require_spine_id] = require_gpu_num
            # 根据x_solution返回每个server占用的gpu数量
            server_occupy_gpuNum_map = {}
            for it in name_list_x_i:
                server_occupy_gpuNum_map[int(it)] = int(require_gpu_within_server*round(xnum_i_solution[it]))
            new_oxc_whether_valid = [2 for i in range(self.oxc_num)]
            for oxc_id in self.oxc_leaf_spine_map:
                for leaf_id in self.oxc_leaf_spine_map[oxc_id]:
                    spine_id = self.oxc_leaf_spine_map[oxc_id][leaf_id]
                    if spine_id == require_spine_id:
                        new_oxc_whether_valid[oxc_id] -= 1
                    if(new_oxc_whether_valid[oxc_id]<0):
                        print(server_occupy_gpuNum_map)
                        print(m.getAttr('X',y_j))
                        print(oxc_whether_valid[oxc_id])
                        print(new_oxc_whether_valid[oxc_id])
                        print("fuck: "+str(oxc_id))
                        print(job_allocated_oxc_spine_link[oxc_id])
                    assert new_oxc_whether_valid[oxc_id]>=0
            return True, server_occupy_gpuNum_map
        else:
            # raise Exception("something wrong4 in gurobi solver")
            return False, None



    def check_valid_and_get_valid_i_k(self):
        valid_i_k = {}
        for i in range(self.leaf_num):
            for k in range(self.oxc_num):
                if i not in valid_i_k:
                    valid_i_k[i] = {}
                valid_i_k[i][k] = 1
        valid_j_k = {}
        for j in range(self.spine_num):
            for k in range(self.oxc_num):
                if j not in valid_j_k:
                    valid_j_k[j] = {}
                valid_j_k[j][k] = 2
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                for k in range(self.oxc_num):
                    if self.oxc_leaf_spine_map[k][i] == j:
                        valid_i_k[i][k] -= 1
                        valid_j_k[j][k] -= 1
                        assert valid_i_k[i][k]>=0
                        if valid_j_k[j][k]<0:
                            print("fuck ",k,j,i)
                        assert valid_j_k[j][k]>=0
        return valid_i_k, valid_j_k

    