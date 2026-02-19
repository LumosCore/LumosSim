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
import numpy as np
from math import ceil
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
        self.spine_ocs_link_num = int(self.port_per_spine/oxc_num)

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


    # 根据leaf_to_spine_map通信需求，以及self.oxc_leaf_spine_map 连接关系，确定新的连接关系
    def update_oxc_leaf_spine_map(self,sim_time=-1): 
        leaf_to_spine_map = self.leaf_to_spine_map
        m = gurobipy.Model("Clos solution")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 120)
        m.setParam('ConcurrentMIP', 64)
        name_list_Z_ijk = []
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                for k in range(self.oxc_num):
                    name_list_Z_ijk.append(str(i)+'_'+str(j)+'_'+str(k))
        Z_i_j_k = {}
        for it in name_list_Z_ijk:
            Z_i_j_k[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='Z_i_j_k')
        obj_val = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=4096,name='obj')
        m.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        m.update()
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                m.addConstr(gurobipy.quicksum( Z_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num)) >= leaf_to_spine_map[i][j])
        for i in range(self.leaf_num):
            for k in range(self.oxc_num):
                m.addConstr(gurobipy.quicksum( Z_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] for j in range(self.spine_num) )<=1)
        for j in range(self.spine_num):
            for k in range(self.oxc_num):
                m.addConstr(gurobipy.quicksum( Z_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) )<=2)
        m.addConstr(obj_val >= gurobipy.quicksum(Z_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) for j in range(self.spine_num) for k in range(self.oxc_num) if  j != self.oxc_leaf_spine_map [k][i] and -1 != self.oxc_leaf_spine_map [k][i] ) 
        + gurobipy.quicksum(1-Z_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) for j in range(self.spine_num) for k in range(self.oxc_num) if  j == self.oxc_leaf_spine_map [k][i] and -1 != self.oxc_leaf_spine_map [k][i] ))

        # 开始执行
        m.update()
        m.optimize()
        # 记录运行结果
        if m.status == gurobipy.GRB.Status.OPTIMAL:
            print(int(obj_val.X))
            valid= True
            Z_i_j_k_solution = m.getAttr('X',Z_i_j_k)
            result = {}
            job_allocated_oxc_spine_link = {}
            for oxc_id in range(self.oxc_num):
                if oxc_id not in result:
                    result[oxc_id] = {}
                for leaf_id in range(self.leaf_num):
                    result[oxc_id][leaf_id] = -1
            for it in name_list_Z_ijk:
                divid_index = it.split("_")
                for id in range(len(divid_index)):
                    divid_index[id] = int(divid_index[id])
                if int(round(Z_i_j_k_solution[it])) == 1:
                    if(int(divid_index[2]) not in result):
                        result[int(divid_index[2])] = {}
                    result[int(divid_index[2])][int(divid_index[0])] = int(divid_index[1])
            for oxc_id in range(self.oxc_num):
                for leaf_id in range(self.leaf_num):
                    if -1 != self.oxc_leaf_spine_map [oxc_id][leaf_id] and result [oxc_id][leaf_id] != self.oxc_leaf_spine_map [oxc_id][leaf_id]:
                        #print(oxc_id, leaf_id, self.oxc_leaf_spine_map [oxc_id][leaf_id], result [oxc_id][leaf_id])
                        valid = False
                    if result [oxc_id][leaf_id] != self.oxc_leaf_spine_map [oxc_id][leaf_id]:
                        if oxc_id not in job_allocated_oxc_spine_link:
                            job_allocated_oxc_spine_link[oxc_id] = {}
                        job_allocated_oxc_spine_link[oxc_id][leaf_id] = result [oxc_id][leaf_id]
 
            return result, job_allocated_oxc_spine_link
        else:
            self.print_leaf_to_spine_map()
            raise Exception("something wrong3 in gurobi solver")
            return False

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

    def find_valid_leaf_pair(self, require_gpu_num, leaf_remain_empt_server_list):
        Z_leafId_oxcId = {}
        for oxc_id in self.oxc_leaf_spine_map:
            for leaf_id in self.oxc_leaf_spine_map[oxc_id]:
                spine_id = self.oxc_leaf_spine_map[oxc_id][leaf_id]
                if spine_id == -1:
                    Z_leafId_oxcId[(leaf_id,oxc_id)] = 0
                else:
                    Z_leafId_oxcId[(leaf_id,oxc_id)] = 1
        
        m = gurobipy.Model("SpineStrategy solution")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 300)
        name_list_x1_i = []
        name_list_z_i_t = []
        name_list_z_t = []
        for i in range(self.leaf_num):
            name_list_x1_i.append(str(i))
        for t in range(self.oxc_num):
            name_list_z_t.append(str(t))
        for i in range(self.leaf_num):
            for t in range(self.oxc_num):
                name_list_z_i_t.append(str(i)+'_'+str(t))

        x1_i = {}
        for it in name_list_x1_i:
            x1_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x1_i')
        z_t = {}
        for it in name_list_z_t:
            z_t[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='z_t')
        z_i_t = {}
        for it in name_list_z_i_t:
            z_i_t[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='z_i_t')

        require_server_num_per_leaf = int(require_gpu_num/2/self.gpu_per_server)

        obj_val = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
        m.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        m.update()

        # 线性化条件
        # link between leaf and OXC
        for i in range(self.leaf_num):
            for t in range(self.oxc_num):
                z_i_t[str(i)+'_'+str(t)] <= Z_leafId_oxcId[(i,t)]*x1_i[str(i)]
                z_i_t[str(i)+'_'+str(t)] <= Z_leafId_oxcId[(i,t)]*z_t[str(t)]

        # used port num is 2 for each to be used OXC
        for t in range(self.oxc_num):
            m.addConstr(gurobipy.quicksum(z_i_t[str(i)+'_'+str(t)] for i in range(self.leaf_num))  == 2*z_t[str(t)])

        # for each to be used server, the link to be used aiming at OCSes should be gpu_num/2
        for i in range(self.leaf_num):
            m.addConstr(gurobipy.quicksum(z_i_t[str(i)+'_'+str(t)] for t in range(self.oxc_num))  == int(require_gpu_num/2)*x1_i[str(i)])

        # used_server_per_leaf is valid
        for i in range(self.leaf_num):
            m.addConstr(leaf_remain_empt_server_list[i] >= x1_i[str(i)]*require_server_num_per_leaf)

        # used leaf num is 2
        m.addConstr(gurobipy.quicksum(x1_i[str(i)] for i in range(self.leaf_num)) == 2)
            
        # set obj
        m.addConstr(obj_val >= gurobipy.quicksum( x1_i[str(i)]*leaf_remain_empt_server_list[i] for i in range(self.leaf_num) ))

        # 开始执行
        m.update()
        m.optimize()
        # 记录运行结果
        if m.status == gurobipy.GRB.Status.OPTIMAL:
            x_i_solution = m.getAttr('X',x1_i)
            x_res = []
            for it in name_list_x1_i:
                if round(x_i_solution[it]) == 1:
                    x_res.append(int(it))
            
            return True, x_res
        else:
            # raise Exception("something wrong4 in gurobi solver")
            return False, None
            
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
        #m.setObjective(obj_val, gurobipy.GRB.MAXIMIZE)
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
            m.addConstr(self.gpu_per_server*xnum_i[str(i)]>=x_i[str(i)])
        
        for i in range(self.server_num):
            m.addConstr(self.gpu_per_server*xnum_i[str(i)]== gurobipy.quicksum( z_i_j[str(i)+'_'+str(j)] for j in range(self.oxc_num)) )

        for i in range(self.server_num):
            for j in range(self.oxc_num):
                m.addConstr(z_i_j[str(i)+'_'+str(j)] <= x_i[str(i)])
                m.addConstr(z_i_j[str(i)+'_'+str(j)] <= y_j[str(j)])
                
        m.addConstr(gurobipy.quicksum(x_i[str(i)]*self.gpu_per_server for i in range(self.server_num)) == require_gpu_num)
        # 目标函数
        # y1 = {}
        # y2 = {}
        # for i in range(self.server_num):
        #     y1[i] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='y1'+str(i))
        #     y2[i] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='y2'+str(i))
        #     m.addConstr(z - self.gpu_per_server*xnum_i[str(i)]<=10*(1-y1[i]))
        #     m.addConstr(x_i[str(i)]<=10*(1-y2[i]))
        #     m.addConstr(x_i[str(i)]>=-10*(1-y2[i]))
        #     m.addConstr(y1[i]+y2[i]>=1)
        leaf_num_k = {}
        for k in range(self.leaf_num):
            leaf_num_k[str(k)] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='leaf_num'+str(k))
        for i in range(self.server_num):
            m.addConstr(leaf_num_k[str(int(i/self.server_per_leaf))]>=x_i[str(i)])
        # m.addConstr(obj_val <= 1000*z + 100*gurobipy.quicksum(x_i[str(i)] for i in range(self.server_num)) - gurobipy.quicksum( leaf_num_k[str(k)] for k in range(self.leaf_num)))
        # #m.addConstr(obj_val <=  2000 - gurobipy.quicksum( leaf_num_k[str(k)] * leaf_remain_empt_server_list[k] for k in range(self.leaf_num)))
        # m.addConstr(obj_val <=  2000 + 100*gurobipy.quicksum(x_i[str(i)] for i in range(self.server_num)) - gurobipy.quicksum( leaf_num_k[str(k)] * leaf_remain_empt_server_list[k] for k in range(self.leaf_num)))
        m.addConstr(obj_val >= gurobipy.quicksum( leaf_num_k[str(k)] for k in range(self.leaf_num)))
        #m.addConstr(obj_val >= gurobipy.quicksum( x_i[str(i)]*server_remain_gpuNum_map[i] for i in range(self.server_num)))
        #m.addConstr(obj_val >= gurobipy.quicksum( leaf_num_k[str(k)]*(5-leaf_remain_empt_server_list[k]) for k in range(self.leaf_num)))

        # 开始执行
        m.update()
        m.optimize()
        # 记录运行结果
        if m.status == gurobipy.GRB.Status.OPTIMAL:
            Z_i_j_solution = m.getAttr('X',z_i_j)
            xnum_i_solution = m.getAttr('X',xnum_i)
            print("fuckfuck", gurobipy.quicksum(x_i[str(i)] for i in range(self.server_num)).getValue(), require_gpu_num)
            assert(round(gurobipy.quicksum(x_i[str(i)] for i in range(self.server_num)).getValue())*self.gpu_per_server == require_gpu_num)
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
                server_occupy_gpuNum_map[int(it)] = int(self.gpu_per_server*round(xnum_i_solution[it]))
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


     # 在情况a中，选择好leaf和spine后就可以更新leaf_to_spine_map，然后调用整数规划配置oxc
    def update_leaf_to_spine_map_according_to_chosen_leaf_and_spine_for_large_job(self, chosen_leaf_id_num_list, choosed_spine_index_list, gpu_num, job_allocated_oxc_spine_link,job_used_spine_port_num_pair):
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
        # 此时oxc更新很简单，因为leaf为空，因此leaf到所有oxc必为空，因此只需要在spine中选择合适的oxc然后连接leaf和spine即可,首先针对spine选择oxc
        spine_chosen_oxc_list_map = {}
        for chosen_spine_id in choosed_spine_index_list:
            chosen_oxc_id_list = []
            need_chosen_oxc = int(gpu_num/len(choosed_spine_index_list))
            job_used_spine_port_num_pair[chosen_spine_id] = need_chosen_oxc
            oxc_whether_valid = [2 for i in range(self.oxc_num)]
            for oxc_id in self.oxc_leaf_spine_map:
                for leaf_id in self.oxc_leaf_spine_map[oxc_id]:
                    spine_id = self.oxc_leaf_spine_map[oxc_id][leaf_id]
                    if spine_id == chosen_spine_id:
                        oxc_whether_valid[oxc_id] -= 1

            for oxc_id in range(self.oxc_num):
                if len(chosen_oxc_id_list)<need_chosen_oxc and oxc_whether_valid[oxc_id]>0:
                    for inputtime in range(oxc_whether_valid[oxc_id]):
                        chosen_oxc_id_list.append(oxc_id)

            spine_chosen_oxc_list_map[chosen_spine_id] = chosen_oxc_id_list
            assert len(chosen_oxc_id_list)==need_chosen_oxc
        # 然后遍历temp_leaf_to_spine_map，对于每一个leaf-spine的需求，选择oxc
        for leaf_id in temp_leaf_to_spine_map:
            for spine_id in temp_leaf_to_spine_map[leaf_id]:
                for chosenTime in range(temp_leaf_to_spine_map[leaf_id][spine_id]):
                    have_chosen = False
                    #print(temp_leaf_to_spine_map[leaf_id][spine_id], leaf_id, spine_id, spine_chosen_oxc_list_map[spine_id])
                    for oxc_id_index, oxc_id in  enumerate(spine_chosen_oxc_list_map[spine_id]):
                        if not have_chosen and self.oxc_leaf_spine_map[oxc_id][leaf_id] == -1:
                            #print("debug update_leaf_to_spine_map_according_to_chosen")
                            self.oxc_leaf_spine_map[oxc_id][leaf_id] = spine_id
                            del(spine_chosen_oxc_list_map[spine_id][oxc_id_index])
                            if oxc_id not in job_allocated_oxc_spine_link:
                                job_allocated_oxc_spine_link[oxc_id] = {}
                            if leaf_id not in job_allocated_oxc_spine_link[oxc_id]:
                                job_allocated_oxc_spine_link[oxc_id][leaf_id] = {}
                            job_allocated_oxc_spine_link[oxc_id][leaf_id] = spine_id
                            have_chosen = True
                    if have_chosen == False:
                        print("debug oxc_leaf_spine_map",self.oxc_leaf_spine_map)
                    assert have_chosen
        return self.oxc_leaf_spine_map , temp_leaf_to_spine_map

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

        
    def update_leaf_to_spine_map_according_to_gpu_size(self, require_gpu_size, leaf_remain_empt_server_list, spine_remain_empt_port_list, require_leaf_num, require_spine_num, job_id):
        # find sub virtual leaf and virtual spine
        leaf_index_sub_leaf_index_list_map = {}
        spine_max_sub_spine_num_map = {}
        sub_leaf_require_server_num = max(1,int(require_spine_num/self.gpu_per_server))
        for leaf_id in range(len(leaf_remain_empt_server_list)):
            num_sub_leaf = int(leaf_remain_empt_server_list[leaf_id]/sub_leaf_require_server_num)
            leaf_index_sub_leaf_index_list_map[leaf_id] = [i for i in range(num_sub_leaf)] # attention that some leaf have no subleaf, which means can not form a clos
        for spine_id in range(len(spine_remain_empt_port_list)):
            num_sub_spine = int(spine_remain_empt_port_list[spine_id]/require_leaf_num)
            spine_max_sub_spine_num_map[spine_id] = num_sub_spine
        # find C_i_j_k
        C_i_j_k = {}
        valid_i_k, valid_j_k = self.check_valid_and_get_valid_i_k()
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                for k in range(self.oxc_num):
                    C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] = valid_i_k[i][k]*valid_j_k[j][k]
        # set parameter
        model = gurobipy.Model("SpineStrategy solution2")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 180)
        model.setParam("MIPGap", 0.8) 
        #model.setParam("MIPFocus",1)
        model.setParam('ConcurrentMIP', 1)
        name_list_s_i = []
        name_list_x_i_a = []
        name_list_y_j = []
        name_list_c_i_a_j_k = []
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                name_list_s_i.append(str(i))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            name_list_c_i_a_j_k.append(str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                name_list_x_i_a.append(str(i)+'_'+str(a))
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                name_list_y_j.append(str(j))
        s_i = {}
        x_i_a = {}
        y_j = {}
        c_i_a_j_k = {}
        spine_used_id = {}
        leaf_used_id = {}
        for it in name_list_s_i:
            s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
        for it in name_list_x_i_a:
            x_i_a[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i_a')
        for it in name_list_y_j:
            y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=spine_max_sub_spine_num_map[int(it)],name='y_j')
        for it in name_list_c_i_a_j_k:
            c_i_a_j_k[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='c_i_a_j_k')
        for spine_id in range(self.spine_num):
            spine_used_id[str(spine_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='spine_used_id')
        for leaf_id in range(self.leaf_num):
            leaf_used_id[str(leaf_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='leaf_used_id')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=4097,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        #
        # set constraint1, virtual leaf number constraint
        model.addConstr(gurobipy.quicksum( x_i_a[it] for it in name_list_x_i_a) == require_leaf_num)
        # set constraint2, virtual spine number constraint
        model.addConstr(gurobipy.quicksum( y_j[it] for it in name_list_y_j) == require_spine_num)
        # set constraint3,  uplink of each leaf meet locality constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr(gurobipy.quicksum(x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) == self.gpu_per_server*s_i[str(i)] )
        # set constraint4, uplink of each leaf to ocs <=1
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i] for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) <= valid_i_k[i][k] )
        # set constraint5, downlink of each spine to ocs <=2
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) <= valid_j_k[j][k] )
        # set constraint6,  uplink of each leaf meet 哦测试link constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i]  ) <= C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] )
        # set constraint7,  uplink of each virtual leaf meet Clos requirement
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) == x_i_a[str(i)+'_'+str(a)]*require_spine_num)
        # # set constraint8,  dowmlink of each virtual spine meet Clos requirement
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) == y_j[str(j)]*require_leaf_num ) 
        # set constraint 9, each virtual leaf/spine only has 1 link
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= x_i_a[str(i)+'_'+str(a)] )
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= y_j[str(j)] )
        # set constraint 10
        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num)   for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] if spine_max_sub_spine_num_map[j]>0) == require_leaf_num*require_spine_num)
        # set obj
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                 for i in range(self.leaf_num):
                    for a in leaf_index_sub_leaf_index_list_map[i]:
                        for k in range(self.oxc_num):
                            model.addConstr( spine_used_id[str(j)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])
                            model.addConstr( leaf_used_id[str(i)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])    
        model.addConstr( obj_val>=  gurobipy.quicksum( spine_used_id[str(j)]*spine_remain_empt_port_list[j] for j in range(self.spine_num))
                                + gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i]*self.gpu_per_server for i in range(self.leaf_num)))
        # # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            c_i_a_j_k_solution = model.getAttr('X', c_i_a_j_k)
            x_i_a_solution = model.getAttr('X', x_i_a)
            y_j_b_solution = model.getAttr('X', y_j)
            s_i_solution = model.getAttr('X', s_i)
            leaf_occupy_gpu_num_map = {}
            spine_occupy_port_num_map = {}
            job_oxc_leaf_spine_map = {}
            job_leaf_to_spine_map = {}
            temp_conn_num = 0
            for i in range(self.leaf_num):
                for a in leaf_index_sub_leaf_index_list_map[i]:
                    for j in range(self.spine_num):
                        if spine_max_sub_spine_num_map[j]>0:
                            for k in range(self.oxc_num):
                                if round(c_i_a_j_k_solution[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)]):
                                    temp_conn_num+=1
                                    if k not in job_oxc_leaf_spine_map:
                                        job_oxc_leaf_spine_map[k] = {}
                                    assert i not in job_oxc_leaf_spine_map[k]
                                    job_oxc_leaf_spine_map[k][i] = j
                                    if self.oxc_leaf_spine_map[k][i] != -1:
                                        print(valid_i_k[i][k],valid_j_k[j][k], C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)])
                                    assert self.oxc_leaf_spine_map[k][i] == -1
                                    self.oxc_leaf_spine_map[k][i] = j
                                    self.leaf_to_spine_map[i][j] += 1

                                    if i not in job_leaf_to_spine_map:
                                        job_leaf_to_spine_map[i] = {}
                                    if j not in job_leaf_to_spine_map[i]:
                                        job_leaf_to_spine_map[i][j] = 0
                                    job_leaf_to_spine_map[i][j] += 1
                                    
                                    if i not in leaf_occupy_gpu_num_map:
                                        leaf_occupy_gpu_num_map[i] = 0
                                    leaf_occupy_gpu_num_map[i] += 1
                                    
                                    if j not in spine_occupy_port_num_map:
                                        spine_occupy_port_num_map[j] = 0
                                    spine_occupy_port_num_map[j] += 1
            print("return result", require_leaf_num, require_spine_num, temp_conn_num)
            print(leaf_occupy_gpu_num_map)
            print(spine_occupy_port_num_map)
            self.check_valid_and_get_valid_i_k()
            return True, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, False

        else:
            print("fuck0", require_leaf_num, require_spine_num)
            print(leaf_remain_empt_server_list, sum(leaf_remain_empt_server_list))
            print(spine_remain_empt_port_list, sum(spine_remain_empt_port_list))
            print(sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list]))
            print(sum([int(i/require_leaf_num) for i in spine_remain_empt_port_list]))
            # self.print_leaf_to_spine_map()
            if sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list])>=require_leaf_num:
                 return False, None, None, None, None, False                                                                                                                                                           
            return False, None, None, None, None, True
        
    def choose_leaf_and_gpu_resource(self, require_gpu_size, leaf_remain_empt_server_list, require_leaf_num, require_spine_num):
        leaf_index_sub_leaf_index_list_map = {}
        sub_leaf_require_server_num = max(1,int(require_spine_num/self.gpu_per_server))
        for leaf_id in range(len(leaf_remain_empt_server_list)):
            num_sub_leaf = int(leaf_remain_empt_server_list[leaf_id]/sub_leaf_require_server_num)
            leaf_index_sub_leaf_index_list_map[leaf_id] = [i for i in range(num_sub_leaf)] # attention that some leaf have no subleaf, which means can not form a clos
        # find idle link num for each leaf
        C_i_k = {}
        valid_i_k, valid_j_k = self.check_valid_and_get_valid_i_k()
        for i in range(self.leaf_num):
            for k in range(self.oxc_num):
                C_i_k[str(i)+'_'+str(k)] = valid_i_k[i][k]
        # set parameter
        model = gurobipy.Model("SpineStrategy solution2")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 180)
        model.setParam("MIPGap", 0.8) 
        #model.setParam("MIPFocus",1)
        model.setParam('ConcurrentMIP', 1)
        
        name_list_s_i = []
        name_list_x_i_a = []
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                name_list_s_i.append(str(i))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                name_list_x_i_a.append(str(i)+'_'+str(a))
        s_i = {}
        x_i_a = {}
        leaf_used_id = {}
        for it in name_list_s_i:
            s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
        for it in name_list_x_i_a:
            x_i_a[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i_a')
        for leaf_id in range(self.leaf_num):
            leaf_used_id[str(leaf_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='leaf_used_id')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40970,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        #
        # set constraint1, virtual leaf number constraint
        model.addConstr(gurobipy.quicksum( x_i_a[it] for it in name_list_x_i_a) == require_leaf_num)
        # set constraint2,  uplink of each leaf meet locality constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr(gurobipy.quicksum(x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) == self.gpu_per_server*s_i[str(i)] )
        # set constraint 3
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr( leaf_used_id[str(i)]*self.gpu_per_leaf >= s_i[str(i)]*self.gpu_per_server )    
        # set constraint 4
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr( gurobipy.quicksum(s_i[str(i)]*self.gpu_per_server for i in range(self.leaf_num) if leaf_index_sub_leaf_index_list_map[i]!=[]) == require_gpu_size)    


        # set obj
        model.addConstr( obj_val>=  gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i] for i in range(self.leaf_num) ))
        # # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            x_i_a_solution = model.getAttr('X', x_i_a)
            s_i_solution = model.getAttr('X', s_i)
            leaf_occupy_gpu_num_map = {}
            for i in range(self.leaf_num):
                if leaf_index_sub_leaf_index_list_map[i]!=[]:
                    leaf_occupy_gpu_num_map[i] = int(s_i_solution[str(i)]*self.gpu_per_server)
            # print("find valid leaf",require_leaf_num,require_spine_num) 
            # for key in leaf_occupy_gpu_num_map:
            #     if round(leaf_occupy_gpu_num_map[key]) != 0:
            #         print(key,round(leaf_occupy_gpu_num_map[key]))
            return True, leaf_occupy_gpu_num_map

        else:
            print("no valid leaf", require_leaf_num, require_spine_num)     
            print("idle server in each leaf",leaf_remain_empt_server_list)                                                                                                                                                 
            return False, []
        
    def find_valid_network_new(self, require_gpu_size, leaf_to_use_server_num, spine_remain_empt_port_list, require_leaf_num, require_spine_num, job_id, leaf_remain_empt_gpu_list, banned_spine_list):
        leaf_index_sub_leaf_index_list_map = {}
        sub_leaf_require_server_num = max(1,int(require_spine_num/self.gpu_per_server))
        for leaf_id in range(self.leaf_num):
            num_sub_leaf = 0
            if leaf_id in leaf_to_use_server_num:
                num_sub_leaf = int(leaf_to_use_server_num[leaf_id]/self.gpu_per_server/sub_leaf_require_server_num)
            leaf_index_sub_leaf_index_list_map[leaf_id] = [i for i in range(num_sub_leaf)]
        spine_max_sub_spine_num_map = {}
        for spine_id in range(len(spine_remain_empt_port_list)):
            num_sub_spine = int(spine_remain_empt_port_list[spine_id]/require_leaf_num)
            if spine_id in banned_spine_list:
                num_sub_spine = 0
            spine_max_sub_spine_num_map[spine_id] = num_sub_spine
        # find C_i_j_k
        C_i_j_k = {}
        valid_i_k, valid_j_k = self.check_valid_and_get_valid_i_k()
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for j in range(self.spine_num):
                    for k in range(self.oxc_num):
                        C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] = valid_i_k[i][k]*valid_j_k[j][k]
                        # print(str(i)+'_'+str(j)+'_'+str(k), C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)])
        # set parameter
        model = gurobipy.Model("SpineStrategy solution2")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 180)
        model.setParam("MIPGap", 0.8) 
        #model.setParam("MIPFocus",1)
        model.setParam('ConcurrentMIP', 1)
        name_list_s_i = []
        name_list_x_i_a = []
        name_list_y_j = []
        name_list_c_i_a_j_k = []
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                name_list_s_i.append(str(i))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            name_list_c_i_a_j_k.append(str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                name_list_x_i_a.append(str(i)+'_'+str(a))
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                name_list_y_j.append(str(j))
        s_i = {}
        x_i_a = {}
        y_j = {}
        c_i_a_j_k = {}
        spine_used_id = {}
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                s_i[str(i)] = leaf_to_use_server_num[i]
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                x_i_a[str(i)+'_'+str(a)] = 1
        print(s_i)
        print(x_i_a)
        print(spine_max_sub_spine_num_map)
        for it in name_list_y_j:
            y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=spine_max_sub_spine_num_map[int(it)],name='y_j')
        for it in name_list_c_i_a_j_k:
            c_i_a_j_k[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='c_i_a_j_k')
        for spine_id in range(self.spine_num):
            spine_used_id[str(spine_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='spine_used_id')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40970,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        #
        # set constraint1, virtual leaf number constraint
        # model.addConstr(gurobipy.quicksum( x_i_a[it] for it in name_list_x_i_a) == require_leaf_num)
        # set constraint2, virtual spine number constraint
        model.addConstr(gurobipy.quicksum( y_j[it] for it in name_list_y_j) == require_spine_num)
        # set constraint3,  uplink of each leaf meet locality constraint
        # for i in range(self.leaf_num):
        #     if leaf_index_sub_leaf_index_list_map[i]!=[]:
        #         model.addConstr(gurobipy.quicksum(x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) == self.gpu_per_server*s_i[str(i)] )
        # set constraint4, uplink of each leaf to ocs <=1
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i] for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) <= valid_i_k[i][k] )
        # set constraint5, downlink of each spine to ocs <=2
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) <= valid_j_k[j][k] )
        # # set constraint6,  uplink of each leaf meet 哦测试link constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i]  ) <= C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] )
        # set constraint7,  uplink of each virtual leaf meet Clos requirement
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) == x_i_a[str(i)+'_'+str(a)]*require_spine_num)
        # # # set constraint8,  dowmlink of each virtual spine meet Clos requirement
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) == y_j[str(j)]*require_leaf_num ) 
        # set constraint 9, each virtual leaf/spine only has 1 link
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= x_i_a[str(i)+'_'+str(a)] )
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= y_j[str(j)] )
        # set constraint 10
        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num)   for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] if spine_max_sub_spine_num_map[j]>0) == require_leaf_num*require_spine_num)
        
        # set constraint 11
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                 for i in range(self.leaf_num):
                    for a in leaf_index_sub_leaf_index_list_map[i]:
                        for k in range(self.oxc_num):
                            model.addConstr( spine_used_id[str(j)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])  


        # set obj
        model.addConstr( obj_val>=  gurobipy.quicksum( spine_used_id[str(j)]*spine_remain_empt_port_list[j] for j in range(self.spine_num)))
        #                         + gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i]*self.gpu_per_server for i in range(self.leaf_num)))
        # # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            c_i_a_j_k_solution = model.getAttr('X', c_i_a_j_k)
            y_j_b_solution = model.getAttr('X', y_j)
            leaf_occupy_gpu_num_map = {}
            leaf_remain_gpu_num_map = {}
            spine_occupy_port_num_map = {}
            job_oxc_leaf_spine_map = {}
            job_leaf_to_spine_map = {}
            temp_conn_num = 0
            for i in range(self.leaf_num):
                for a in leaf_index_sub_leaf_index_list_map[i]:
                    for j in range(self.spine_num):
                        if spine_max_sub_spine_num_map[j]>0:
                            for k in range(self.oxc_num):
                                if round(c_i_a_j_k_solution[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)]):
                                    temp_conn_num+=1
                                    if k not in job_oxc_leaf_spine_map:
                                        job_oxc_leaf_spine_map[k] = {}
                                    assert i not in job_oxc_leaf_spine_map[k]
                                    job_oxc_leaf_spine_map[k][i] = j
                                    if self.oxc_leaf_spine_map[k][i] != -1:
                                        print(valid_i_k[i][k],valid_j_k[j][k], C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)])
                                    assert self.oxc_leaf_spine_map[k][i] == -1
                                    self.oxc_leaf_spine_map[k][i] = j
                                    self.leaf_to_spine_map[i][j] += 1

                                    if i not in job_leaf_to_spine_map:
                                        job_leaf_to_spine_map[i] = {}
                                    if j not in job_leaf_to_spine_map[i]:
                                        job_leaf_to_spine_map[i][j] = 0
                                    job_leaf_to_spine_map[i][j] += 1
                                    
                                    if i not in leaf_occupy_gpu_num_map:
                                        leaf_occupy_gpu_num_map[i] = 0
                                    leaf_occupy_gpu_num_map[i] += 1
                                    
                                    if j not in spine_occupy_port_num_map:
                                        spine_occupy_port_num_map[j] = 0
                                    spine_occupy_port_num_map[j] += 1
            print("return result", require_leaf_num, require_spine_num, temp_conn_num)
            # print(leaf_occupy_gpu_num_map)
            # print(spine_occupy_port_num_map)
            self.check_valid_and_get_valid_i_k()
            return True, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, False, leaf_remain_gpu_num_map

        else:
            print("fuck.0.5", require_leaf_num, require_spine_num)
            # print(s_i)
            # print(x_i_a)
            # print(spine_remain_empt_port_list, sum(spine_remain_empt_port_list))
            # print(sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list]))
            # print(sum([int(i/require_leaf_num) for i in spine_remain_empt_port_list]))
            # # self.print_leaf_to_spine_map()
            # if sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list])>=require_leaf_num:
            #      return False, None, None, None, None, False  , None                                                                                                                                                         
            return False, None, None, None, None, True, None
        
    def find_valid_gpu_new(self, require_gpu_size, leaf_remain_empt_server_list, spine_remain_empt_port_list, require_leaf_num, require_spine_num, job_id, leaf_remain_empt_gpu_list):
        # 找到最接近的幂数t，仍然寻找m*n=t的vclos，但是要求选择的leaf包含的空余GPU之和为z-t
        remain_chosen_gpu_num = require_gpu_size - require_leaf_num*require_spine_num
        leaf_index_sub_leaf_index_list_map = {}
        spine_max_sub_spine_num_map = {}
        sub_leaf_require_server_num = max(1,int(require_spine_num/self.gpu_per_server))
        for leaf_id in range(len(leaf_remain_empt_server_list)):
            num_sub_leaf = int(leaf_remain_empt_server_list[leaf_id]/sub_leaf_require_server_num)
            leaf_index_sub_leaf_index_list_map[leaf_id] = [i for i in range(num_sub_leaf)] # attention that some leaf have no subleaf, which means can not form a clos
        for spine_id in range(len(spine_remain_empt_port_list)):
            num_sub_spine = int(spine_remain_empt_port_list[spine_id]/require_leaf_num)
            spine_max_sub_spine_num_map[spine_id] = num_sub_spine
        # find C_i_j_k
        C_i_j_k = {}
        valid_i_k, valid_j_k = self.check_valid_and_get_valid_i_k()
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                for k in range(self.oxc_num):
                    C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] = valid_i_k[i][k]*valid_j_k[j][k]
        # set parameter
        model = gurobipy.Model("SpineStrategy solution2")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 180)
        model.setParam("MIPGap", 0.8) 
        #model.setParam("MIPFocus",1)
        model.setParam('ConcurrentMIP', 1)
        name_list_s_i = []
        name_list_x_i_a = []
        name_list_y_j = []
        name_list_c_i_a_j_k = []
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                name_list_s_i.append(str(i))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            name_list_c_i_a_j_k.append(str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                name_list_x_i_a.append(str(i)+'_'+str(a))
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                name_list_y_j.append(str(j))
        s_i = {}
        x_i_a = {}
        y_j = {}
        c_i_a_j_k = {}
        spine_used_id = {}
        leaf_used_id = {}
        for it in name_list_s_i:
            s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
        for it in name_list_x_i_a:
            x_i_a[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i_a')
        for it in name_list_y_j:
            y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=spine_max_sub_spine_num_map[int(it)],name='y_j')
        for it in name_list_c_i_a_j_k:
            c_i_a_j_k[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='c_i_a_j_k')
        for spine_id in range(self.spine_num):
            spine_used_id[str(spine_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='spine_used_id')
        for leaf_id in range(self.leaf_num):
            leaf_used_id[str(leaf_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='leaf_used_id')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40970,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        #
        # set constraint1, virtual leaf number constraint
        model.addConstr(gurobipy.quicksum( x_i_a[it] for it in name_list_x_i_a) == require_leaf_num)
        # set constraint2, virtual spine number constraint
        model.addConstr(gurobipy.quicksum( y_j[it] for it in name_list_y_j) == require_spine_num)
        # set constraint3,  uplink of each leaf meet locality constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr(gurobipy.quicksum(x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) == self.gpu_per_server*s_i[str(i)] )
        # set constraint4, uplink of each leaf to ocs <=1
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i] for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) <= valid_i_k[i][k] )
        # set constraint5, downlink of each spine to ocs <=2
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) <= valid_j_k[j][k] )
        # set constraint6,  uplink of each leaf meet 哦测试link constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i]  ) <= C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] )
        # set constraint7,  uplink of each virtual leaf meet Clos requirement
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) == x_i_a[str(i)+'_'+str(a)]*require_spine_num)
        # # set constraint8,  dowmlink of each virtual spine meet Clos requirement
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) == y_j[str(j)]*require_leaf_num ) 
        # set constraint 9, each virtual leaf/spine only has 1 link
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= x_i_a[str(i)+'_'+str(a)] )
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= y_j[str(j)] )
        # set constraint 10
        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num)   for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] if spine_max_sub_spine_num_map[j]>0) == require_leaf_num*require_spine_num)
        # set constraint 11
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                 for i in range(self.leaf_num):
                    for a in leaf_index_sub_leaf_index_list_map[i]:
                        for k in range(self.oxc_num):
                            model.addConstr( spine_used_id[str(j)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])
                            model.addConstr( leaf_used_id[str(i)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])    


        # set obj
        model.addConstr( obj_val>=  gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i]*self.gpu_per_server for i in range(self.leaf_num) ))
        # # model.addConstr( obj_val>=  gurobipy.quicksum( spine_used_id[str(j)]*spine_remain_empt_port_list[j] for j in range(self.spine_num))
        #                         + gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i]*self.gpu_per_server for i in range(self.leaf_num)))
        # # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            c_i_a_j_k_solution = model.getAttr('X', c_i_a_j_k)
            x_i_a_solution = model.getAttr('X', x_i_a)
            y_j_b_solution = model.getAttr('X', y_j)
            s_i_solution = model.getAttr('X', s_i)
            leaf_occupy_gpu_num_map = {}
            leaf_remain_gpu_num_map = {}
            spine_occupy_port_num_map = {}
            job_oxc_leaf_spine_map = {}
            job_leaf_to_spine_map = {}
            temp_conn_num = 0
            for i in range(self.leaf_num):
                for a in leaf_index_sub_leaf_index_list_map[i]:
                    for j in range(self.spine_num):
                        if spine_max_sub_spine_num_map[j]>0:
                            for k in range(self.oxc_num):
                                if round(c_i_a_j_k_solution[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)]):
                                    temp_conn_num+=1
                                    if k not in job_oxc_leaf_spine_map:
                                        job_oxc_leaf_spine_map[k] = {}
                                    assert i not in job_oxc_leaf_spine_map[k]
                                    job_oxc_leaf_spine_map[k][i] = j
                                    if self.oxc_leaf_spine_map[k][i] != -1:
                                        print(valid_i_k[i][k],valid_j_k[j][k], C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)])
                                    assert self.oxc_leaf_spine_map[k][i] == -1
                                    self.oxc_leaf_spine_map[k][i] = j
                                    self.leaf_to_spine_map[i][j] += 1

                                    if i not in job_leaf_to_spine_map:
                                        job_leaf_to_spine_map[i] = {}
                                    if j not in job_leaf_to_spine_map[i]:
                                        job_leaf_to_spine_map[i][j] = 0
                                    job_leaf_to_spine_map[i][j] += 1
                                    
                                    if i not in leaf_occupy_gpu_num_map:
                                        leaf_occupy_gpu_num_map[i] = 0
                                    leaf_occupy_gpu_num_map[i] += 1
                                    
                                    if j not in spine_occupy_port_num_map:
                                        spine_occupy_port_num_map[j] = 0
                                    spine_occupy_port_num_map[j] += 1
            print("return result", require_leaf_num, require_spine_num, temp_conn_num)
            print(leaf_occupy_gpu_num_map)
            print(spine_occupy_port_num_map)
            self.check_valid_and_get_valid_i_k()
            return True, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, False, leaf_remain_gpu_num_map

        else:
            print("fuck.0.5", require_leaf_num, require_spine_num)
            print(leaf_remain_empt_server_list, sum(leaf_remain_empt_server_list))
            print(spine_remain_empt_port_list, sum(spine_remain_empt_port_list))
            print(sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list]))
            print(sum([int(i/require_leaf_num) for i in spine_remain_empt_port_list]))
            # self.print_leaf_to_spine_map()
            if sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list])>=require_leaf_num:
                 return False, None, None, None, None, False  , None                                                                                                                                                         
            return False, None, None, None, None, True, None
        
    def find_valid_gpu_for_no_pow2_task(self, require_gpu_size, leaf_remain_empt_server_list, spine_remain_empt_port_list, require_leaf_num, require_spine_num, job_id, leaf_remain_empt_gpu_list):
        # 找到最接近的幂数t，仍然寻找m*n=t的vclos，但是要求选择的leaf包含的空余GPU之和为z-t
        remain_chosen_gpu_num = require_gpu_size - require_leaf_num*require_spine_num
        leaf_index_sub_leaf_index_list_map = {}
        spine_max_sub_spine_num_map = {}
        sub_leaf_require_server_num = max(1,int(require_spine_num/self.gpu_per_server))
        for leaf_id in range(len(leaf_remain_empt_server_list)):
            num_sub_leaf = int(leaf_remain_empt_server_list[leaf_id]/sub_leaf_require_server_num)
            leaf_index_sub_leaf_index_list_map[leaf_id] = [i for i in range(num_sub_leaf)] # attention that some leaf have no subleaf, which means can not form a clos
        for spine_id in range(len(spine_remain_empt_port_list)):
            num_sub_spine = int(spine_remain_empt_port_list[spine_id]/require_leaf_num)
            spine_max_sub_spine_num_map[spine_id] = num_sub_spine
        # find C_i_j_k
        C_i_j_k = {}
        valid_i_k, valid_j_k = self.check_valid_and_get_valid_i_k()
        for i in range(self.leaf_num):
            for j in range(self.spine_num):
                for k in range(self.oxc_num):
                    C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] = valid_i_k[i][k]*valid_j_k[j][k]
        # set parameter
        model = gurobipy.Model("SpineStrategy solution2")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 180)
        model.setParam("MIPGap", 0.8) 
        #model.setParam("MIPFocus",1)
        model.setParam('ConcurrentMIP', 1)
        name_list_s_i = []
        name_list_x_i_a = []
        name_list_y_j = []
        name_list_c_i_a_j_k = []
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                name_list_s_i.append(str(i))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            name_list_c_i_a_j_k.append(str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k))
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                name_list_x_i_a.append(str(i)+'_'+str(a))
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                name_list_y_j.append(str(j))
        s_i = {}
        x_i_a = {}
        y_j = {}
        c_i_a_j_k = {}
        spine_used_id = {}
        leaf_used_id = {}
        x_r_i = {}
        temp_x_r_i = {}
        for it in name_list_s_i:
            s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
            # print(leaf_remain_empt_gpu_list)
            # print(int(it))
            # print(leaf_remain_empt_server_list)
            # print(leaf_remain_empt_gpu_list[int(it)])
            x_r_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_gpu_list[int(it)],name='x_r_i')
            temp_x_r_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=int(leaf_remain_empt_gpu_list[int(it)]/2),name='x_r_i')
        for it in name_list_x_i_a:
            x_i_a[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i_a')
        for it in name_list_y_j:
            y_j[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=spine_max_sub_spine_num_map[int(it)],name='y_j')
        for it in name_list_c_i_a_j_k:
            c_i_a_j_k[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='c_i_a_j_k')
        for spine_id in range(self.spine_num):
            spine_used_id[str(spine_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='spine_used_id')
        for leaf_id in range(self.leaf_num):
            leaf_used_id[str(leaf_id)] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='leaf_used_id')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=4097,name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        #
        # set constraint1, virtual leaf number constraint
        model.addConstr(gurobipy.quicksum( x_i_a[it] for it in name_list_x_i_a) == require_leaf_num)
        # set constraint2, virtual spine number constraint
        model.addConstr(gurobipy.quicksum( y_j[it] for it in name_list_y_j) == require_spine_num)
        # set constraint3,  uplink of each leaf meet locality constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr(gurobipy.quicksum(x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) == self.gpu_per_server*s_i[str(i)] )
        # set constraint4, uplink of each leaf to ocs <=1
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i] for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) <= valid_i_k[i][k] )
        # set constraint5, downlink of each spine to ocs <=2
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                for k in range(self.oxc_num):
                    model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) <= valid_j_k[j][k] )
        # set constraint6,  uplink of each leaf meet 哦测试link constraint
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        for k in range(self.oxc_num):
                            model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i]  ) <= C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] )
        # set constraint7,  uplink of each virtual leaf meet Clos requirement
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num) if spine_max_sub_spine_num_map[j]>0) == x_i_a[str(i)+'_'+str(a)]*require_spine_num)
        # # set constraint8,  dowmlink of each virtual spine meet Clos requirement
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) == y_j[str(j)]*require_leaf_num ) 
        # set constraint 9, each virtual leaf/spine only has 1 link
        for i in range(self.leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(self.spine_num):
                    if spine_max_sub_spine_num_map[j]>0:
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= x_i_a[str(i)+'_'+str(a)] )
                        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) ) <= y_j[str(j)] )
        # set constraint 10
        model.addConstr( gurobipy.quicksum( c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)] for k in range(self.oxc_num) for j in range(self.spine_num)   for i in range(self.leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] if spine_max_sub_spine_num_map[j]>0) == require_leaf_num*require_spine_num)
        # set constraint 11
        model.addConstr( gurobipy.quicksum( x_r_i[str(i)] for i in range(self.leaf_num) if leaf_index_sub_leaf_index_list_map[i]!=[]) == remain_chosen_gpu_num )
        for i in range(self.leaf_num):
            if leaf_index_sub_leaf_index_list_map[i]!=[]:
                model.addConstr(  x_r_i[str(i)] == 2*temp_x_r_i[str(i)] )
                model.addConstr(  x_r_i[str(i)] + gurobipy.quicksum( x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) <= leaf_remain_empt_gpu_list[i] )
                model.addConstr(  x_r_i[str(i)] <= gurobipy.quicksum( x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]))
        for j in range(self.spine_num):
            if spine_max_sub_spine_num_map[j]>0:
                 for i in range(self.leaf_num):
                    for a in leaf_index_sub_leaf_index_list_map[i]:
                        for k in range(self.oxc_num):
                            model.addConstr( spine_used_id[str(j)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])
                            model.addConstr( leaf_used_id[str(i)] >=  c_i_a_j_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)])    
        # for i in range(self.leaf_num):
        #     if leaf_index_sub_leaf_index_list_map[i]!=[]:
        #         model.addConstr(  x_r_i[str(i)] >= leaf_used_id[str(i)])

        # set obj
        # model.addConstr( obj_val>=  gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i]*self.gpu_per_server for i in range(self.leaf_num) ))
        # # model.addConstr( obj_val>=  gurobipy.quicksum( spine_used_id[str(j)]*spine_remain_empt_port_list[j] for j in range(self.spine_num))
        #                         + gurobipy.quicksum( leaf_used_id[str(i)]*leaf_remain_empt_server_list[i]*self.gpu_per_server for i in range(self.leaf_num)))
        # # 开始执行
        model.update()
        model.optimize()
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            c_i_a_j_k_solution = model.getAttr('X', c_i_a_j_k)
            x_i_a_solution = model.getAttr('X', x_i_a)
            y_j_b_solution = model.getAttr('X', y_j)
            s_i_solution = model.getAttr('X', s_i)
            x_r_i_solution = model.getAttr('X', x_r_i)
            leaf_occupy_gpu_num_map = {}
            leaf_remain_gpu_num_map = {}
            spine_occupy_port_num_map = {}
            job_oxc_leaf_spine_map = {}
            job_leaf_to_spine_map = {}
            temp_conn_num = 0
            for i in range(self.leaf_num):
                if leaf_index_sub_leaf_index_list_map[i]!=[]:
                    if round(x_r_i_solution[str(i)])>0:
                        if i not in leaf_remain_gpu_num_map:
                            leaf_remain_gpu_num_map[i] = 0
                        leaf_remain_gpu_num_map[i] += round(x_r_i_solution[str(i)])
                for a in leaf_index_sub_leaf_index_list_map[i]:
                    for j in range(self.spine_num):
                        if spine_max_sub_spine_num_map[j]>0:
                            for k in range(self.oxc_num):
                                if round(c_i_a_j_k_solution[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(k)]):
                                    temp_conn_num+=1
                                    if k not in job_oxc_leaf_spine_map:
                                        job_oxc_leaf_spine_map[k] = {}
                                    assert i not in job_oxc_leaf_spine_map[k]
                                    job_oxc_leaf_spine_map[k][i] = j
                                    if self.oxc_leaf_spine_map[k][i] != -1:
                                        print(valid_i_k[i][k],valid_j_k[j][k], C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)])
                                    assert self.oxc_leaf_spine_map[k][i] == -1
                                    self.oxc_leaf_spine_map[k][i] = j
                                    self.leaf_to_spine_map[i][j] += 1

                                    if i not in job_leaf_to_spine_map:
                                        job_leaf_to_spine_map[i] = {}
                                    if j not in job_leaf_to_spine_map[i]:
                                        job_leaf_to_spine_map[i][j] = 0
                                    job_leaf_to_spine_map[i][j] += 1
                                    
                                    if i not in leaf_occupy_gpu_num_map:
                                        leaf_occupy_gpu_num_map[i] = 0
                                    leaf_occupy_gpu_num_map[i] += 1
                                    
                                    if j not in spine_occupy_port_num_map:
                                        spine_occupy_port_num_map[j] = 0
                                    spine_occupy_port_num_map[j] += 1
            print("return result", require_leaf_num, require_spine_num, temp_conn_num)
            print(leaf_occupy_gpu_num_map)
            print(spine_occupy_port_num_map)
            self.check_valid_and_get_valid_i_k()
            return True, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, False, leaf_remain_gpu_num_map

        else:
            print("fuck.0.5", require_leaf_num, require_spine_num)
            print(leaf_remain_empt_server_list, sum(leaf_remain_empt_server_list))
            print(spine_remain_empt_port_list, sum(spine_remain_empt_port_list))
            print(sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list]))
            print(sum([int(i/require_leaf_num) for i in spine_remain_empt_port_list]))
            # self.print_leaf_to_spine_map()
            if sum([int(i*self.gpu_per_server/require_spine_num) for i in leaf_remain_empt_server_list])>=require_leaf_num:
                 return False, None, None, None, None, False  , None                                                                                                                                                         
            return False, None, None, None, None, True, None
        
    def find_valid_gpu_for_specific_spine_new(self, require_gpu_num, require_spine_id, server_remain_gpuNum_map,job_allocated_oxc_spine_link, used_spine_port_num_pair, leaf_remain_empt_server_list):
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
        
        solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
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
            x_i[it] = solver.IntVar(0.0, 1.0, 'x_i'+str(it))
        y_j = {}
        for it in name_list_y_j:
            y_j[it] = solver.IntVar(0.0, 2.0, 'y_j'+str(it))
        z_i_j = {}
        for it in name_list_z_i_j:
            z_i_j[it] =solver.IntVar(0.0, 1.0, 'z_i_j'+str(it))
        xnum_i = {}
        for it in name_list_x_i:
            xnum_i[it] = solver.IntVar(0.0, 1.0, 'xnum_i'+str(it))
        objective = solver.Objective()
        
        
        #constraint 1
        solver.Add(y_j[str(0)]<=oxc_whether_valid[0])
        solver.Add(y_j[str(1)]<=oxc_whether_valid[1])
        solver.Add(y_j[str(2)]<=oxc_whether_valid[2])
        solver.Add(y_j[str(3)]<=oxc_whether_valid[3])
        
        # #constraint2
        solver.Add(z_i_j["0_0"]+z_i_j["1_0"]+z_i_j["2_0"]+z_i_j["3_0"]+z_i_j["4_0"]+z_i_j["5_0"]+z_i_j["6_0"]+z_i_j["7_0"]+z_i_j["8_0"]+z_i_j["9_0"]+z_i_j["10_0"]+z_i_j["11_0"]+z_i_j["12_0"]+z_i_j["13_0"]+z_i_j["14_0"]+z_i_j["15_0"] <=  y_j[str(0)] )
        solver.Add(z_i_j["0_1"]+z_i_j["1_1"]+z_i_j["2_1"]+z_i_j["3_1"]+z_i_j["4_1"]+z_i_j["5_1"]+z_i_j["6_1"]+z_i_j["7_1"]+z_i_j["8_1"]+z_i_j["9_1"]+z_i_j["10_1"]+z_i_j["11_1"]+z_i_j["12_1"]+z_i_j["13_1"]+z_i_j["14_1"]+z_i_j["15_1"] <=  y_j[str(1)] )
        solver.Add(z_i_j["0_2"]+z_i_j["1_2"]+z_i_j["2_2"]+z_i_j["3_2"]+z_i_j["4_2"]+z_i_j["5_2"]+z_i_j["6_2"]+z_i_j["7_2"]+z_i_j["8_2"]+z_i_j["9_2"]+z_i_j["10_2"]+z_i_j["11_2"]+z_i_j["12_2"]+z_i_j["13_2"]+z_i_j["14_2"]+z_i_j["15_2"] <=  y_j[str(2)] )
        solver.Add(z_i_j["0_3"]+z_i_j["1_3"]+z_i_j["2_3"]+z_i_j["3_3"]+z_i_j["4_3"]+z_i_j["5_3"]+z_i_j["6_3"]+z_i_j["7_3"]+z_i_j["8_3"]+z_i_j["9_3"]+z_i_j["10_3"]+z_i_j["11_3"]+z_i_j["12_3"]+z_i_j["13_3"]+z_i_j["14_3"]+z_i_j["15_3"] <=  y_j[str(3)] )

        #constraint3
        solver.Add(z_i_j["0_0"] + Z_leafId_oxcId[(0,0)] <=1)
        solver.Add(z_i_j["0_1"] + Z_leafId_oxcId[(0,1)] <=1)
        solver.Add(z_i_j["0_2"] + Z_leafId_oxcId[(0,2)] <=1)
        solver.Add(z_i_j["0_3"] + Z_leafId_oxcId[(0,3)] <=1)
        solver.Add(z_i_j["1_0"] + Z_leafId_oxcId[(1,0)] <=1)
        solver.Add(z_i_j["1_1"] + Z_leafId_oxcId[(1,1)] <=1)
        solver.Add(z_i_j["1_2"] + Z_leafId_oxcId[(1,2)] <=1)
        solver.Add(z_i_j["1_3"] + Z_leafId_oxcId[(1,3)] <=1)
        solver.Add(z_i_j["2_0"] + Z_leafId_oxcId[(2,0)] <=1)
        solver.Add(z_i_j["2_1"] + Z_leafId_oxcId[(2,1)] <=1)
        solver.Add(z_i_j["2_2"] + Z_leafId_oxcId[(2,2)] <=1)
        solver.Add(z_i_j["2_3"] + Z_leafId_oxcId[(2,3)] <=1)
        solver.Add(z_i_j["3_0"] + Z_leafId_oxcId[(3,0)] <=1)
        solver.Add(z_i_j["3_1"] + Z_leafId_oxcId[(3,1)] <=1)
        solver.Add(z_i_j["3_2"] + Z_leafId_oxcId[(3,2)] <=1)
        solver.Add(z_i_j["3_3"] + Z_leafId_oxcId[(3,3)] <=1)
        solver.Add(z_i_j["4_0"] + Z_leafId_oxcId[(4,0)] <=1)
        solver.Add(z_i_j["4_1"] + Z_leafId_oxcId[(4,1)] <=1)
        solver.Add(z_i_j["4_2"] + Z_leafId_oxcId[(4,2)] <=1)
        solver.Add(z_i_j["4_3"] + Z_leafId_oxcId[(4,3)] <=1)
        solver.Add(z_i_j["5_0"] + Z_leafId_oxcId[(5,0)] <=1)
        solver.Add(z_i_j["5_1"] + Z_leafId_oxcId[(5,1)] <=1)
        solver.Add(z_i_j["5_2"] + Z_leafId_oxcId[(5,2)] <=1)
        solver.Add(z_i_j["5_3"] + Z_leafId_oxcId[(5,3)] <=1)
        solver.Add(z_i_j["6_0"] + Z_leafId_oxcId[(6,0)] <=1)
        solver.Add(z_i_j["6_1"] + Z_leafId_oxcId[(6,1)] <=1)
        solver.Add(z_i_j["6_2"] + Z_leafId_oxcId[(6,2)] <=1)
        solver.Add(z_i_j["6_3"] + Z_leafId_oxcId[(6,3)] <=1)
        solver.Add(z_i_j["7_0"] + Z_leafId_oxcId[(7,0)] <=1)
        solver.Add(z_i_j["7_1"] + Z_leafId_oxcId[(7,1)] <=1)
        solver.Add(z_i_j["7_2"] + Z_leafId_oxcId[(7,2)] <=1)
        solver.Add(z_i_j["7_3"] + Z_leafId_oxcId[(7,3)] <=1)
        solver.Add(z_i_j["8_0"] + Z_leafId_oxcId[(8,0)] <=1)
        solver.Add(z_i_j["8_1"] + Z_leafId_oxcId[(8,1)] <=1)
        solver.Add(z_i_j["8_2"] + Z_leafId_oxcId[(8,2)] <=1)
        solver.Add(z_i_j["8_3"] + Z_leafId_oxcId[(8,3)] <=1)
        solver.Add(z_i_j["9_0"] + Z_leafId_oxcId[(9,0)] <=1)
        solver.Add(z_i_j["9_1"] + Z_leafId_oxcId[(9,1)] <=1)
        solver.Add(z_i_j["9_2"] + Z_leafId_oxcId[(9,2)] <=1)
        solver.Add(z_i_j["9_3"] + Z_leafId_oxcId[(9,3)] <=1)
        solver.Add(z_i_j["10_0"] + Z_leafId_oxcId[(10,0)] <=1)
        solver.Add(z_i_j["10_1"] + Z_leafId_oxcId[(10,1)] <=1)
        solver.Add(z_i_j["10_2"] + Z_leafId_oxcId[(10,2)] <=1)
        solver.Add(z_i_j["10_3"] + Z_leafId_oxcId[(10,3)] <=1)
        solver.Add(z_i_j["11_0"] + Z_leafId_oxcId[(11,0)] <=1)
        solver.Add(z_i_j["11_1"] + Z_leafId_oxcId[(11,1)] <=1)
        solver.Add(z_i_j["11_2"] + Z_leafId_oxcId[(11,2)] <=1)
        solver.Add(z_i_j["11_3"] + Z_leafId_oxcId[(11,3)] <=1)
        solver.Add(z_i_j["12_0"] + Z_leafId_oxcId[(12,0)] <=1)
        solver.Add(z_i_j["12_1"] + Z_leafId_oxcId[(12,1)] <=1)
        solver.Add(z_i_j["12_2"] + Z_leafId_oxcId[(12,2)] <=1)
        solver.Add(z_i_j["12_3"] + Z_leafId_oxcId[(12,3)] <=1)
        solver.Add(z_i_j["13_0"] + Z_leafId_oxcId[(13,0)] <=1)
        solver.Add(z_i_j["13_1"] + Z_leafId_oxcId[(13,1)] <=1)
        solver.Add(z_i_j["13_2"] + Z_leafId_oxcId[(13,2)] <=1)
        solver.Add(z_i_j["13_3"] + Z_leafId_oxcId[(13,3)] <=1)
        solver.Add(z_i_j["14_0"] + Z_leafId_oxcId[(14,0)] <=1)
        solver.Add(z_i_j["14_1"] + Z_leafId_oxcId[(14,1)] <=1)
        solver.Add(z_i_j["14_2"] + Z_leafId_oxcId[(14,2)] <=1)
        solver.Add(z_i_j["14_3"] + Z_leafId_oxcId[(14,3)] <=1)
        solver.Add(z_i_j["15_0"] + Z_leafId_oxcId[(15,0)] <=1)
        solver.Add(z_i_j["15_1"] + Z_leafId_oxcId[(15,1)] <=1)
        solver.Add(z_i_j["15_2"] + Z_leafId_oxcId[(15,2)] <=1)
        solver.Add(z_i_j["15_3"] + Z_leafId_oxcId[(15,3)] <=1)
                   
        #constraint4
        solver.Add(z_i_j["0_0"]+z_i_j["0_1"]+z_i_j["0_2"]+z_i_j["0_3"] <= server_remain_gpuNum_map[0] )
        solver.Add(z_i_j["1_0"]+z_i_j["1_1"]+z_i_j["1_2"]+z_i_j["1_3"] <= server_remain_gpuNum_map[1] )
        solver.Add(z_i_j["2_0"]+z_i_j["2_1"]+z_i_j["2_2"]+z_i_j["2_3"] <= server_remain_gpuNum_map[2] )
        solver.Add(z_i_j["3_0"]+z_i_j["3_1"]+z_i_j["3_2"]+z_i_j["3_3"] <= server_remain_gpuNum_map[3] )
        solver.Add(z_i_j["4_0"]+z_i_j["4_1"]+z_i_j["4_2"]+z_i_j["4_3"] <= server_remain_gpuNum_map[4] )
        solver.Add(z_i_j["5_0"]+z_i_j["5_1"]+z_i_j["5_2"]+z_i_j["5_3"] <= server_remain_gpuNum_map[5] )
        solver.Add(z_i_j["6_0"]+z_i_j["6_1"]+z_i_j["6_2"]+z_i_j["6_3"] <= server_remain_gpuNum_map[6] )
        solver.Add(z_i_j["7_0"]+z_i_j["7_1"]+z_i_j["7_2"]+z_i_j["7_3"] <= server_remain_gpuNum_map[7] )
        solver.Add(z_i_j["8_0"]+z_i_j["8_1"]+z_i_j["8_2"]+z_i_j["8_3"] <= server_remain_gpuNum_map[8] )
        solver.Add(z_i_j["9_0"]+z_i_j["9_1"]+z_i_j["9_2"]+z_i_j["9_3"] <= server_remain_gpuNum_map[9] )
        solver.Add(z_i_j["10_0"]+z_i_j["10_1"]+z_i_j["10_2"]+z_i_j["10_3"] <= server_remain_gpuNum_map[3] )
        solver.Add(z_i_j["11_0"]+z_i_j["11_1"]+z_i_j["11_2"]+z_i_j["11_3"] <= server_remain_gpuNum_map[11] )
        solver.Add(z_i_j["12_0"]+z_i_j["12_1"]+z_i_j["12_2"]+z_i_j["12_3"] <= server_remain_gpuNum_map[12] )
        solver.Add(z_i_j["13_0"]+z_i_j["13_1"]+z_i_j["13_2"]+z_i_j["13_3"] <= server_remain_gpuNum_map[13] )
        solver.Add(z_i_j["14_0"]+z_i_j["14_1"]+z_i_j["14_2"]+z_i_j["14_3"] <= server_remain_gpuNum_map[14] )
        solver.Add(z_i_j["15_0"]+z_i_j["15_1"]+z_i_j["15_2"]+z_i_j["15_3"] <= server_remain_gpuNum_map[15] )
        
        #constraint5
        solver.Add(z_i_j["0_0"]+z_i_j["0_1"]+z_i_j["0_2"]+z_i_j["0_3"]+z_i_j["1_0"]+z_i_j["1_1"]+z_i_j["1_2"]+z_i_j["1_3"]+z_i_j["2_0"]+z_i_j["2_1"]+z_i_j["2_2"]+z_i_j["2_3"]+z_i_j["3_0"]+z_i_j["3_1"]+z_i_j["3_2"]+z_i_j["3_3"]+z_i_j["4_0"]+z_i_j["4_1"]+z_i_j["4_2"]+z_i_j["4_3"]+z_i_j["5_0"]+z_i_j["5_1"]+z_i_j["5_2"]+z_i_j["5_3"]+z_i_j["6_0"]+z_i_j["6_1"]+z_i_j["6_2"]+z_i_j["6_3"]+z_i_j["7_0"]+z_i_j["7_1"]+z_i_j["7_2"]+z_i_j["7_3"]+z_i_j["8_0"]+z_i_j["8_1"]+z_i_j["8_2"]+z_i_j["8_3"]+z_i_j["9_0"]+z_i_j["9_1"]+z_i_j["9_2"]+z_i_j["9_3"]+z_i_j["10_0"]+z_i_j["10_1"]+z_i_j["10_2"]+z_i_j["10_3"]+z_i_j["11_0"]+z_i_j["11_1"]+z_i_j["11_2"]+z_i_j["11_3"]+z_i_j["12_0"]+z_i_j["12_1"]+z_i_j["12_2"]+z_i_j["12_3"]+z_i_j["13_0"]+z_i_j["13_1"]+z_i_j["13_2"]+z_i_j["13_3"]+z_i_j["14_0"]+z_i_j["14_1"]+z_i_j["14_2"]+z_i_j["14_3"]+z_i_j["15_0"]+z_i_j["15_1"]+z_i_j["15_2"]+z_i_j["15_3"]== require_gpu_num)
        #constraint6
        solver.Add(y_j["0"]+y_j["1"]+y_j["2"]+y_j['3'] == require_gpu_num)

        #constraint7
        solver.Add(self.gpu_per_server*xnum_i[str(0)]>=x_i[str(0)])
        solver.Add(self.gpu_per_server*xnum_i[str(1)]>=x_i[str(1)])
        solver.Add(self.gpu_per_server*xnum_i[str(2)]>=x_i[str(2)])
        solver.Add(self.gpu_per_server*xnum_i[str(3)]>=x_i[str(3)])
        solver.Add(self.gpu_per_server*xnum_i[str(4)]>=x_i[str(4)])
        solver.Add(self.gpu_per_server*xnum_i[str(5)]>=x_i[str(5)])
        solver.Add(self.gpu_per_server*xnum_i[str(6)]>=x_i[str(6)])
        solver.Add(self.gpu_per_server*xnum_i[str(7)]>=x_i[str(7)])
        solver.Add(self.gpu_per_server*xnum_i[str(8)]>=x_i[str(8)])
        solver.Add(self.gpu_per_server*xnum_i[str(9)]>=x_i[str(9)])
        solver.Add(self.gpu_per_server*xnum_i[str(10)]>=x_i[str(10)])
        solver.Add(self.gpu_per_server*xnum_i[str(11)]>=x_i[str(11)])
        solver.Add(self.gpu_per_server*xnum_i[str(12)]>=x_i[str(12)])
        solver.Add(self.gpu_per_server*xnum_i[str(13)]>=x_i[str(13)])
        solver.Add(self.gpu_per_server*xnum_i[str(14)]>=x_i[str(14)])
        solver.Add(self.gpu_per_server*xnum_i[str(15)]>=x_i[str(15)])
                
        #constraint8
        solver.Add(self.gpu_per_server*xnum_i[str(0)]==(z_i_j["0_0"]+z_i_j["0_1"]+z_i_j["0_2"]+z_i_j["0_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(1)]==(z_i_j["1_0"]+z_i_j["1_1"]+z_i_j["1_2"]+z_i_j["1_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(2)]==(z_i_j["2_0"]+z_i_j["2_1"]+z_i_j["2_2"]+z_i_j["2_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(3)]==(z_i_j["3_0"]+z_i_j["3_1"]+z_i_j["3_2"]+z_i_j["3_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(4)]==(z_i_j["4_0"]+z_i_j["4_1"]+z_i_j["4_2"]+z_i_j["4_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(5)]==(z_i_j["5_0"]+z_i_j["5_1"]+z_i_j["5_2"]+z_i_j["5_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(6)]==(z_i_j["6_0"]+z_i_j["6_1"]+z_i_j["6_2"]+z_i_j["6_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(7)]==(z_i_j["7_0"]+z_i_j["7_1"]+z_i_j["7_2"]+z_i_j["7_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(8)]==(z_i_j["8_0"]+z_i_j["8_1"]+z_i_j["8_2"]+z_i_j["8_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(9)]==(z_i_j["9_0"]+z_i_j["9_1"]+z_i_j["9_2"]+z_i_j["9_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(10)]==(z_i_j["10_0"]+z_i_j["10_1"]+z_i_j["10_2"]+z_i_j["10_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(11)]==(z_i_j["11_0"]+z_i_j["11_1"]+z_i_j["11_2"]+z_i_j["11_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(12)]==(z_i_j["12_0"]+z_i_j["12_1"]+z_i_j["12_2"]+z_i_j["12_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(13)]==(z_i_j["13_0"]+z_i_j["13_1"]+z_i_j["13_2"]+z_i_j["13_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(14)]==(z_i_j["14_0"]+z_i_j["14_1"]+z_i_j["14_2"]+z_i_j["14_3"]))
        solver.Add(self.gpu_per_server*xnum_i[str(15)]==(z_i_j["15_0"]+z_i_j["15_1"]+z_i_j["15_2"]+z_i_j["15_3"]))
                   
        #constraint9
        solver.Add(z_i_j["0_0"] <= x_i[str(0)] )
        solver.Add(z_i_j["0_1"] <= x_i[str(0)] )
        solver.Add(z_i_j["0_2"] <= x_i[str(0)] )
        solver.Add(z_i_j["0_3"] <= x_i[str(0)] )
        solver.Add(z_i_j["1_0"] <= x_i[str(1)] )
        solver.Add(z_i_j["1_1"] <= x_i[str(1)] )
        solver.Add(z_i_j["1_2"] <= x_i[str(1)] )
        solver.Add(z_i_j["1_3"] <= x_i[str(1)] )
        solver.Add(z_i_j["2_0"] <= x_i[str(2)] )
        solver.Add(z_i_j["2_1"] <= x_i[str(2)] )
        solver.Add(z_i_j["2_2"] <= x_i[str(2)] )
        solver.Add(z_i_j["2_3"] <= x_i[str(2)] )
        solver.Add(z_i_j["3_0"] <= x_i[str(3)] )
        solver.Add(z_i_j["3_1"] <= x_i[str(3)] )
        solver.Add(z_i_j["3_2"] <= x_i[str(3)] )
        solver.Add(z_i_j["3_3"] <= x_i[str(3)] )
        solver.Add(z_i_j["4_0"] <= x_i[str(4)] )
        solver.Add(z_i_j["4_1"] <= x_i[str(4)] )
        solver.Add(z_i_j["4_2"] <= x_i[str(4)] )
        solver.Add(z_i_j["4_3"] <= x_i[str(4)] )
        solver.Add(z_i_j["5_0"] <= x_i[str(5)] )
        solver.Add(z_i_j["5_1"] <= x_i[str(5)] )
        solver.Add(z_i_j["5_2"] <= x_i[str(5)] )
        solver.Add(z_i_j["5_3"] <= x_i[str(5)] )
        solver.Add(z_i_j["6_0"] <= x_i[str(6)] )
        solver.Add(z_i_j["6_1"] <= x_i[str(6)] )
        solver.Add(z_i_j["6_2"] <= x_i[str(6)] )
        solver.Add(z_i_j["6_3"] <= x_i[str(6)] )
        solver.Add(z_i_j["7_0"] <= x_i[str(7)] )
        solver.Add(z_i_j["7_1"] <= x_i[str(7)] )
        solver.Add(z_i_j["7_2"] <= x_i[str(7)] )
        solver.Add(z_i_j["7_3"] <= x_i[str(7)] )
        solver.Add(z_i_j["8_0"] <= x_i[str(8)] )
        solver.Add(z_i_j["8_1"] <= x_i[str(8)] )
        solver.Add(z_i_j["8_2"] <= x_i[str(8)] )
        solver.Add(z_i_j["8_3"] <= x_i[str(8)] )
        solver.Add(z_i_j["9_0"] <= x_i[str(9)] )
        solver.Add(z_i_j["9_1"] <= x_i[str(9)] )
        solver.Add(z_i_j["9_2"] <= x_i[str(9)] )
        solver.Add(z_i_j["9_3"] <= x_i[str(9)] )
        solver.Add(z_i_j["10_0"] <= x_i[str(10)] )
        solver.Add(z_i_j["10_1"] <= x_i[str(10)] )
        solver.Add(z_i_j["10_2"] <= x_i[str(10)] )
        solver.Add(z_i_j["10_3"] <= x_i[str(10)] )
        solver.Add(z_i_j["11_0"] <= x_i[str(11)] )
        solver.Add(z_i_j["11_1"] <= x_i[str(11)] )
        solver.Add(z_i_j["11_2"] <= x_i[str(11)] )
        solver.Add(z_i_j["11_3"] <= x_i[str(11)] )
        solver.Add(z_i_j["12_0"] <= x_i[str(12)] )
        solver.Add(z_i_j["12_1"] <= x_i[str(12)] )
        solver.Add(z_i_j["12_2"] <= x_i[str(12)] )
        solver.Add(z_i_j["12_3"] <= x_i[str(12)] )
        solver.Add(z_i_j["13_0"] <= x_i[str(13)] )
        solver.Add(z_i_j["13_1"] <= x_i[str(13)] )
        solver.Add(z_i_j["13_2"] <= x_i[str(13)] )
        solver.Add(z_i_j["13_3"] <= x_i[str(13)] )
        solver.Add(z_i_j["14_0"] <= x_i[str(14)] )
        solver.Add(z_i_j["14_1"] <= x_i[str(14)] )
        solver.Add(z_i_j["14_2"] <= x_i[str(14)] )
        solver.Add(z_i_j["14_3"] <= x_i[str(14)] )
        solver.Add(z_i_j["15_0"] <= x_i[str(15)] )
        solver.Add(z_i_j["15_1"] <= x_i[str(15)] )
        solver.Add(z_i_j["15_2"] <= x_i[str(15)] )
        solver.Add(z_i_j["15_3"] <= x_i[str(15)] )
        solver.Add(z_i_j["0_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["0_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["0_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["0_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["1_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["1_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["1_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["1_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["2_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["2_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["2_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["2_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["3_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["3_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["3_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["3_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["4_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["4_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["4_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["4_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["5_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["5_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["5_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["5_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["6_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["6_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["6_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["6_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["7_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["7_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["7_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["7_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["8_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["8_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["8_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["8_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["9_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["9_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["9_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["9_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["10_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["10_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["10_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["10_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["11_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["11_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["11_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["11_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["12_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["12_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["12_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["12_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["13_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["13_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["13_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["13_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["14_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["14_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["14_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["14_3"] <= y_j[str(3)] )
        solver.Add(z_i_j["15_0"] <= y_j[str(0)] )
        solver.Add(z_i_j["15_1"] <= y_j[str(1)] )
        solver.Add(z_i_j["15_2"] <= y_j[str(2)] )
        solver.Add(z_i_j["15_3"] <= y_j[str(3)] )
                
        #constraint10
        solver.Add(x_i[str(0)]*self.gpu_per_server+x_i[str(1)]*self.gpu_per_server+x_i[str(2)]*self.gpu_per_server+x_i[str(3)]*self.gpu_per_server+x_i[str(4)]*self.gpu_per_server+x_i[str(5)]*self.gpu_per_server+x_i[str(6)]*self.gpu_per_server+x_i[str(7)]*self.gpu_per_server+x_i[str(8)]*self.gpu_per_server+x_i[str(9)]*self.gpu_per_server+x_i[str(10)]*self.gpu_per_server+x_i[str(11)]*self.gpu_per_server+x_i[str(12)]*self.gpu_per_server+x_i[str(13)]*self.gpu_per_server+x_i[str(14)]*self.gpu_per_server+x_i[str(15)]*self.gpu_per_server == require_gpu_num)

        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            Z_i_j_solution = {}
            xnum_i_solution = {}
            for it in name_list_z_i_j:
                Z_i_j_solution[it] = z_i_j[it].solution_value()
            for it in name_list_x_i:
                xnum_i_solution[it] = xnum_i[it].solution_value()
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
                server_occupy_gpuNum_map[int(it)] = int(self.gpu_per_server*round(xnum_i_solution[it]))
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