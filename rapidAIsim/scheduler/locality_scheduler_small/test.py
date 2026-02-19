from operator import mod
import gurobipy
import numpy as np

gpu_num =512
server_num = 128
leaf_num = 32
spine_num = 16
oxc_num = 16
gpu_per_server = int(gpu_num/server_num)
gpu_per_leaf = int(gpu_num/leaf_num)
port_per_spine = int(gpu_num/spine_num)

def update_leaf_to_spine_map_according_to_gpu_size(require_gpu_size, leaf_remain_empt_server_list, spine_remain_empt_port_list, require_leaf_num, require_spine_num):
    # find sub virtual leaf and virtual spine
    leaf_index_sub_leaf_index_list_map = {}
    spine_index_sub_spine_index_list_map = {}
    sub_leaf_require_server_num = max(1,int(require_spine_num/gpu_per_server))
    for leaf_id in range(len(leaf_remain_empt_server_list)):
        num_sub_leaf = int(leaf_remain_empt_server_list[leaf_id]/sub_leaf_require_server_num)
        leaf_index_sub_leaf_index_list_map[leaf_id] = [i for i in range(num_sub_leaf)] # attention that some leaf have no subleaf, which means can not form a clos
    for spine_id in range(len(spine_remain_empt_port_list)):
        num_sub_spine = int(spine_remain_empt_port_list[spine_id]/require_leaf_num)
        spine_index_sub_spine_index_list_map[spine_id] = [i for i in range(num_sub_spine)]
    # find C_i_j_k
    C_i_j_k = {}
    valid_i_k = {0: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}, 1: {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 0, 12: 0, 13: 1, 14: 0, 15: 1}, 2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 1, 9: 1, 10: 0, 11: 1, 12: 1, 13: 1, 14: 1, 15: 0}, 4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 5: {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0, 10: 1, 11: 1, 12: 0, 13: 1, 14: 0, 15: 0}, 6: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 7: {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 1, 14: 1, 15: 1}, 8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 9: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 10: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 11: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 12: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 13: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}, 14: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 15: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 16: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 17: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 18: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 19: {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 0, 12: 1, 13: 0, 14: 1, 15: 0}, 20: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 21: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 22: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 23: {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0, 13: 1, 14: 0, 15: 1}, 24: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 25: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 26: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 27: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}, 28: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}, 29: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}, 30: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 31: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}}

    valid_j_k = {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 4: {0: 1, 1: 0, 2: 2, 3: 1, 4: 0, 5: 2, 6: 0, 7: 0, 8: 1, 9: 2, 10: 1, 11: 2, 12: 1, 13: 2, 14: 0, 15: 1}, 5: {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2}, 6: {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2}, 7: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 9: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 10: {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2}, 11: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 12: {0: 1, 1: 2, 2: 0, 3: 2, 4: 1, 5: 0, 6: 2, 7: 0, 8: 2, 9: 0, 10: 2, 11: 0, 12: 0, 13: 2, 14: 1, 15: 1}, 13: {0: 0, 1: 2, 2: 1, 3: 1, 4: 0, 5: 1, 6: 2, 7: 2, 8: 1, 9: 1, 10: 0, 11: 0, 12: 1, 13: 1, 14: 2, 15: 1}, 14: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 15: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}}

    for i in range(leaf_num):
        for j in range(spine_num):
            for k in range(oxc_num):
                C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] = valid_i_k[i][k]*valid_j_k[j][k]
    # set parameter
    model = gurobipy.Model("SpineStrategy solution2")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)
    name_list_s_i = []
    name_list_x_i_a = []
    name_list_y_j_b = []
    name_list_c_i_a_j_b_k = []
    spine_used_id = {}
    for i in range(leaf_num):
        if leaf_index_sub_leaf_index_list_map[i]!=[]:
            name_list_s_i.append(str(i))
    for i in range(leaf_num):
        for a in leaf_index_sub_leaf_index_list_map[i]:
            for j in range(spine_num):
                for b in spine_index_sub_spine_index_list_map[j]:
                    for k in range(oxc_num):
                        name_list_c_i_a_j_b_k.append(str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k))
    for i in range(leaf_num):
        for a in leaf_index_sub_leaf_index_list_map[i]:
            name_list_x_i_a.append(str(i)+'_'+str(a))
    for j in range(spine_num):
        for b in spine_index_sub_spine_index_list_map[j]:
            name_list_y_j_b.append(str(j)+'_'+str(b))
    s_i = {}
    x_i_a = {}
    y_j_b = {}
    c_i_a_j_b_k = {}
    spine_used_id = {}
    for it in name_list_s_i:
        s_i[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=leaf_remain_empt_server_list[int(it)],name='s_i')
    for it in name_list_x_i_a:
        x_i_a[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i_a')
    for it in name_list_y_j_b:
        y_j_b[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='y_j_b')
    for it in name_list_c_i_a_j_b_k:
        c_i_a_j_b_k[it] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='c_i_a_j_b_k')
    for spine_id in range(spine_num):
        spine_used_id[str(spine_id)] =  model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='spine_used_id')
    obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960,name='obj')
    model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
    model.update()

    # set constraint1
    model.addConstr(gurobipy.quicksum( x_i_a[it] for it in name_list_x_i_a) == require_leaf_num)
    # set constraint2
    model.addConstr(gurobipy.quicksum( y_j_b[it] for it in name_list_y_j_b) == require_spine_num)
    # set constraint3
    for i in range(leaf_num):
        if leaf_index_sub_leaf_index_list_map[i]!=[]:
            model.addConstr(gurobipy.quicksum(x_i_a[str(i)+'_'+str(a)]*require_spine_num for a in leaf_index_sub_leaf_index_list_map[i]) == gpu_per_server*s_i[str(i)] )
    # set constraint4
    for i in range(leaf_num):
        if leaf_index_sub_leaf_index_list_map[i]!=[]:
            for k in range(oxc_num):
                model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i] for j in range(spine_num) for b in spine_index_sub_spine_index_list_map[j] ) <= valid_i_k[i][k] )
    # set constraint5
    for j in range(spine_num):
        if spine_index_sub_spine_index_list_map[j]!=[]:
            for k in range(oxc_num):
                model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for i in range(leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] for b in spine_index_sub_spine_index_list_map[j] ) <= valid_j_k[j][k] )
    # set constraint6
    for i in range(leaf_num):
        if leaf_index_sub_leaf_index_list_map[i]!=[]:
            for j in range(spine_num):
                if spine_index_sub_spine_index_list_map[j]!=[]:
                    for k in range(oxc_num):
                        model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for a in leaf_index_sub_leaf_index_list_map[i] for b in spine_index_sub_spine_index_list_map[j] ) <= C_i_j_k[str(i)+'_'+str(j)+'_'+str(k)] )
    # set constraint7
    for i in range(leaf_num):
        for a in leaf_index_sub_leaf_index_list_map[i]:
            model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for k in range(oxc_num) for j in range(spine_num) for b in spine_index_sub_spine_index_list_map[j]  ) == x_i_a[str(i)+'_'+str(a)]*require_spine_num)
    # set constraint8
    for j in range(spine_num):
        for b in spine_index_sub_spine_index_list_map[j]:
            model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for k in range(oxc_num) for i in range(leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) == y_j_b[str(j)+'_'+str(b)]*require_leaf_num ) 
    # set constraint 9
    for i in range(leaf_num):
        for a in leaf_index_sub_leaf_index_list_map[i]:
            for j in range(spine_num):
                for b in spine_index_sub_spine_index_list_map[j]:
                    model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for k in range(oxc_num) ) <= x_i_a[str(i)+'_'+str(a)] )
                    model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for k in range(oxc_num) ) <= y_j_b[str(j)+'_'+str(b)] )
    # set constraint 10
    model.addConstr( gurobipy.quicksum( c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)] for k in range(oxc_num) for j in range(spine_num) for b in spine_index_sub_spine_index_list_map[j]  for i in range(leaf_num) for a in leaf_index_sub_leaf_index_list_map[i] ) == require_leaf_num*require_spine_num)
    # set obj
    for j in range(spine_num):
        for b in spine_index_sub_spine_index_list_map[j]:
            for i in range(leaf_num):
                for a in leaf_index_sub_leaf_index_list_map[i]:
                    for k in range(oxc_num):
                        model.addConstr( spine_used_id[str(j)] >=  c_i_a_j_b_k[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)])
    model.addConstr( obj_val>=  gurobipy.quicksum( spine_used_id[str(j)]*spine_remain_empt_port_list[j] for j in range(spine_num)) )
    # 开始执行
    model.update()
    model.optimize()
    # 记录运行结果
    if model.status == gurobipy.GRB.Status.OPTIMAL:
        c_i_a_j_b_k_solution = model.getAttr('X', c_i_a_j_b_k)
        x_i_a_solution = model.getAttr('X', x_i_a)
        y_j_b_solution = model.getAttr('X', y_j_b)
        s_i_solution = model.getAttr('X', s_i)
        leaf_occupy_gpu_num_map = {}
        spine_occupy_port_num_map = {}
        job_oxc_leaf_spine_map = {}
        job_leaf_to_spine_map = {}
        for i in range(leaf_num):
            for a in leaf_index_sub_leaf_index_list_map[i]:
                for j in range(spine_num):
                    for b in spine_index_sub_spine_index_list_map[j]:
                        for k in range(oxc_num):
                            if int(c_i_a_j_b_k_solution[str(i)+'_'+str(a)+'_'+str(j)+'_'+str(b)+'_'+str(k)]):
                                if k not in job_oxc_leaf_spine_map:
                                    job_oxc_leaf_spine_map[k] = {}
                                assert i not in job_oxc_leaf_spine_map[k]
                                job_oxc_leaf_spine_map[k][i] = j

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
        print("return result", require_leaf_num, require_spine_num)
        print(leaf_occupy_gpu_num_map)
        print(spine_occupy_port_num_map)
        return True, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map

    else:
        print("fuck0", int(require_leaf_num), require_spine_num)
        print(sum([int(i*8/require_spine_num) for i in leaf_remain_empt_server_list]))
        print(sum([int(i/require_leaf_num) for i in spine_remain_empt_port_list]))
        # print_leaf_to_spine_map()
        return False, None, None, None, None

gpu_num = 64    
leaf_remain_empt_server_list=[0, 2, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 4, 4, 0, 4]
spine_remain_empt_port_list=[0, 0, 0, 0, 16, 32, 32, 0, 0, 0, 32, 0, 16, 16, 0, 0]
temp_require_leaf_num=8
temp_require_spine_num=8
update_leaf_to_spine_map_according_to_gpu_size(gpu_num, leaf_remain_empt_server_list, spine_remain_empt_port_list, temp_require_leaf_num, temp_require_spine_num)