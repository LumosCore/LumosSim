import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .Route import RouteTool


def restore_to_integer(N, d_star, R):
    # 参数设置
    # 创建模型
    model = gp.Model("Integer_Recovery")

    # 添加变量
    x = model.addVars(N, N, vtype=GRB.INTEGER, name="x")

    # 设置目标函数
    model.setObjective(gp.quicksum(x[i, j] for i in range(N) for j in range(N)), GRB.MAXIMIZE)

    # 添加约束
    for i in range(N):
        for j in range(N):
            model.addConstr(x[i, j] >= np.floor(d_star[i, j]), name=f"lower_bound_{i}_{j}")
            model.addConstr(x[i, j] <= np.ceil(d_star[i, j]), name=f"upper_bound_{i}_{j}")
            model.addConstr(x[i, j] == x[j, i], name=f"symmetry_{i}_{j}")

    for i in range(N):
        model.addConstr(gp.quicksum(x[j, i] for j in range(N)) <= R[i], name=f"row_sum_{i}")

    # 求解模型
    model.setParam('OutputFlag', False)
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', x)
        return solution


def COUDER(s_matrix, R_c, d_wave, N, u=0, tag="INTEGER"):
    # import pdb;pdb.set_trace()
    from gurobipy import Model, GRB, quicksum
    # 初始化模型
    m = Model("NetworkOptimization")
    # 变量
    f = m.addVars(N, N, N, name="f", lb=0)  # 流量分配变量
    m.setParam('OutputFlag', False)
    if tag == "INTEGER":
        n_matrix = m.addVars(N, N, vtype=GRB.INTEGER, name="n")  # 连接决策变量
    else:
        n_matrix = m.addVars(N, N, name="n")  # 连接决策变量
    # 流量分配和连接决策约束
    for i in range(N):
        m.addConstr(quicksum(n_matrix[i, j] for j in range(N) if j != i) <= R_c[i], name=f"R_limit_{i}")
    for i in range(N):
        for j in range(N):
            m.addConstr(n_matrix[i, j] == n_matrix[j, i], name=f"R_equal_{i}{j}")
    # 目标函数
    u = m.addVar()
    m.setObjective(u, GRB.MAXIMIZE)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i == j or i == k:
                    m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
    # 流量分配总和约束
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(quicksum(f[i, j, k] for k in range(N)) == u, name=f"flow_sum_{i}_{j}")
    # 成本约束
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(quicksum(f[i, jp, j] * d_wave[i, jp] for jp in range(N)) +
                            quicksum(f[ip, j, i] * d_wave[ip, j] for ip in range(N)) <= n_matrix[i, j] * s_matrix[
                                i, j],
                            name=f"cost_{i}_{j}")
    # 模型求解
    # m.setParam('TimeLimit', 500)
    m.setParam('OutputFlag', False)
    m.optimize()
    # f_d = {}
    # for key in f:
    #     f_d[key] = f[key].x / u.x
    # f_d = {(i, j, k): f[i, j, k].x / u.x for i in range(N) for j in range(N) for k in range(N)}
    # 输出结果
    if m.status == GRB.OPTIMAL:
        if tag == "INTEGER":
            n_value = np.zeros((N, N))
            for key in n_matrix:
                n_value[key[0]][key[1]] = n_matrix[key].x
            # return f_d,1 / u.x, n_value
            return {},0,n_value
        else:
            n_value = np.zeros((N, N))
            for key in n_matrix:
                n_value[key[0]][key[1]] = n_matrix[key].x
            solution = restore_to_integer(N, n_value, R_c)
            for key in n_matrix:
                n_value[key[0]][key[1]] = solution[key]
            f, u_now = RouteTool.lp_by_gp(n_value, s_matrix, d_wave, N)
            return f, u_now, n_value

    else:
        print("No optimal solution found")
        return 0, 0
