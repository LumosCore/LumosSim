from gurobipy import *
import numpy as np



# size
LEAF_NUM = -1
SPINE_NUM = -1

# configure cluster size
def config_cluster_size(clos_n, clos_m):
    new_leaf_num = clos_m
    new_spine_num = clos_n

    global LEAF_NUM
    global SPINE_NUM

    LEAF_NUM = new_leaf_num
    SPINE_NUM = new_spine_num



# clos_n is power of 2
def allocate_resources_for_given_mn(gpu_status, spine_ports_status, fixed_clos_n, fixed_clos_m):
    if LEAF_NUM == -1 or SPINE_NUM == -1:
        print("Warning: please configure cluster size first!")
        exit()

    #******** allocation tendency ***********
    spine_remaining_ports = [(len(ss) - sum(ss)) for ss in spine_ports_status]
    # compute weights for allocation tendency
    sorted_si = np.argsort(spine_remaining_ports)
    si_weights = [0 for i in range(SPINE_NUM)]
    for w, si in enumerate(sorted_si):
        si_weights[si] = w
    # compute weights for allocation tendency
    sorted_li = np.argsort(gpu_status)
    li_weights = [0 for i in range(LEAF_NUM)]
    for w, li in enumerate(sorted_li):
        li_weights[li] = w

    spine_allocation_tendency = si_weights
    leaf_allocation_tendency = li_weights
    # ***************************************

    # 定义模型变量
    m = Model("Clos solution")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 10)
    x = [[0 for i in range(SPINE_NUM)] for j in range(LEAF_NUM)]  # current connection
    f = [0 for i in range(LEAF_NUM)]

    clos_m = fixed_clos_m#m.addVar(lb=1, ub=64, vtype=GRB.INTEGER, name='clos_m')
    clos_n = fixed_clos_n#m.addVar(lb=1, ub=32, vtype=GRB.INTEGER, name='clos_n')
    #g = [[m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name='gpu_num'+str(i)+str(j)) for i in range(7)] for j in range(LEAF_NUM)]
    leaf_placement = [m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name='l' + str(i)) for i in range(LEAF_NUM)]
    spine_placement = [m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name='p' + str(i)) for i in range(SPINE_NUM)]
    y = [[m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name='y' + str(i) + str(j)) for i in range(SPINE_NUM)] for j in
         range(LEAF_NUM)]  # variables

    # mapping current network status to X
    for spine_index, ports_status in enumerate(spine_ports_status):
        for port_index, port_status in enumerate(ports_status):
            if port_status > 0:
                a = port_index
                b = spine_index
                x[a][b] = 1

    # mapping current gpu status to f
    for li, free_gpu in enumerate(gpu_status):
        f[li] = free_gpu

    # 添加约束
    # 1. gpu constraint
    '''
    gpu_num_choice = [0, 1, 2, 4, 8, 16, 32]
    constr = 0
    for a in range(LEAF_NUM):
        gpu_num_a = 0
        gpu_size_selected = 0
        for i in range(7):
            gpu_num_a += g[a][i]*gpu_num_choice[i]
            gpu_size_selected += g[a][i]
        m.addConstr(gpu_num_a <= clos_n)
        m.addConstr(gpu_num_a <= leaf_placement[a] * f[a])
        m.addConstr(gpu_size_selected == 1)
        constr += gpu_num_a
    m.addConstr(constr == request_num)
    '''
    for a in range(LEAF_NUM):
        m.addConstr(leaf_placement[a] * clos_n <= f[a])

    # 2. leaf constraint
    constr = 0
    for a in range(LEAF_NUM):
        constr += leaf_placement[a]
    m.addConstr(constr == clos_m)

    # 3. spine constraint
    constr = 0
    for b in range(SPINE_NUM):
        constr += spine_placement[b]
    m.addConstr(constr == clos_n)

    # 4. link constraint
    for a in range(LEAF_NUM):
        for b in range(SPINE_NUM):
            m.addConstr(y[a][b] >= x[a][b])
            # m.addConstr(y[a][b] - x[a][b] == leaf_placement[a]*spine_placement[b])
            m.addConstr((y[a][b] - x[a][b]) <= leaf_placement[a])
            m.addConstr((y[a][b] - x[a][b]) <= spine_placement[b])
            m.addConstr((y[a][b] - x[a][b]) >= leaf_placement[a] + spine_placement[b] - 1)

    # 添加目标函数
    objective = 0
    for a in range(LEAF_NUM):
            objective += (leaf_placement[a] * li_weights[a])

    m.setObjective(objective, GRB.MINIMIZE)#(clos_m * clos_n, GRB.MINIMIZE)

    # 求解
    # print("start solving ...")
    m.optimize()
    print('*' * 60)
    # print('最优值：', m.objVal)
    try:
        res = m.objVal
    except:
        print("No solution")
        return False,-1, -1, [],[],[]

    # translate to port choices
    leaf_choices = []
    spine_choices = []
    gpu_choices = []
    for a in range(LEAF_NUM):
        if leaf_placement[a].x >= 0.9:
            leaf_choices.append(a)
            gpu_choices.append(clos_n)


    for b in range(SPINE_NUM):
        if spine_placement[b].x >= 0.9: spine_choices.append(b)


    # print("Leaf choices:", leaf_choices)
    # print("GPU choices:", gpu_choices)
    # print("Spine choices:", spine_choices)

    return True, clos_n, clos_m, leaf_choices, gpu_choices, spine_choices