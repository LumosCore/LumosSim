import math
import random
from ortools.linear_solver import pywraplp
import numpy as np


def solve(spine_num_per_pod, spine_up_port_num, c_ijt: np.ndarray, u_ijkt: np.ndarray, is_itv = True, releax =False, is_dragon_fly=False, only_facebook = False):
    pod_num = c_ijt.shape[0]
    # print("debug spine_num_per_pod, spine_up_port_num",spine_num_per_pod, spine_up_port_num,pod_num)
    # solver = pywraplp.Solver.CreateSolver('CP_SAT')
    solver = pywraplp.Solver.CreateSolver('Gurobi')
    solver = pywraplp.Solver.CreateSolver('Gurobi')
    solver.SetSolverSpecificParametersAsString("""
        MIPGap=0.1       
        """)
    


    # solver.SetTimeLimit(1000*60)

    # 设置模型变量
    x_ijkt = np.empty((pod_num, pod_num, spine_up_port_num, spine_num_per_pod), dtype=pywraplp.Variable)
    for i in range(pod_num):
        for j in range(pod_num):
            for k in range(spine_up_port_num):
                for t in range(spine_num_per_pod):
                    x_ijkt[i, j, k, t] = solver.IntVar(0, 1, f'x_{i}_{j}_{k}_{t}')
    h_ijkt = np.empty((pod_num, pod_num, spine_up_port_num, spine_num_per_pod), dtype=pywraplp.Variable)
    for i in range(pod_num):
        for j in range(pod_num):
            for k in range(spine_up_port_num):
                for t in range(spine_num_per_pod):
                    h_ijkt[i, j, k, t] = solver.IntVar(-spine_up_port_num, spine_up_port_num, f'h_{i}_{j}_{k}_{t}')
    t_ijt_dis = np.empty((pod_num, pod_num, spine_num_per_pod), dtype=pywraplp.Variable)
    for i in range(pod_num):
        for j in range(pod_num):
            for t in range(spine_num_per_pod):
                t_ijt_dis[i, j, t] = solver.IntVar(0, spine_up_port_num*100*spine_num_per_pod, f't_{i}_{j}_{t}')
    t_it_dis2 = np.empty((pod_num, spine_num_per_pod), dtype=pywraplp.Variable)
    t_jt_dis2 = np.empty((pod_num, spine_num_per_pod), dtype=pywraplp.Variable)
    for i in range(pod_num):
        for t in range(spine_num_per_pod):
            t_it_dis2[i, t] = solver.IntVar(-spine_up_port_num*100*spine_num_per_pod, spine_up_port_num*100*spine_num_per_pod, f't2_{i}_{j}_{t}')
            t_jt_dis2[i, t] = solver.IntVar(-spine_up_port_num*100*spine_num_per_pod, spine_up_port_num*100*spine_num_per_pod, f't3_{i}_{j}_{t}')
    # 设置约束
    # 1. x_ijkt对k和t求和等于c_ijt
    if not releax:
        for i in range(pod_num):
            for j in range(pod_num):
                for t in range(spine_num_per_pod):
                    solver.Add(solver.Sum(x_ijkt[i, j, :, t].ravel().tolist()) == c_ijt[i, j, t])
    else:
        for i in range(pod_num):
            for j in range(pod_num):
                for t in range(spine_num_per_pod):
                    solver.Add(t_ijt_dis[i, j, t] >= c_ijt[i, j, t] - solver.Sum(x_ijkt[i, j, :, t].ravel().tolist()))
    # 1.5 对称
    if not is_itv:
        for i in range(pod_num):
            for j in range(pod_num):
                for k in range(spine_up_port_num):
                    for t in range(spine_num_per_pod):
                        solver.Add(x_ijkt[i, j, k, t] == x_ijkt[j, i, k, t])
    # 2. x_ijkt对j和k求和等于spine_up_port_num
    if not releax:
        for i in range(pod_num):
            for t in range(spine_num_per_pod):
                solver.Add(solver.Sum(x_ijkt[i, :, :, t].ravel().tolist()) == spine_up_port_num)
    else:
        for i in range(pod_num):
            for t in range(spine_num_per_pod):
                solver.Add(spine_up_port_num >= solver.Sum(x_ijkt[i, :, :, t].ravel().tolist()))
                solver.Add(t_it_dis2[i, t] == spine_up_port_num - solver.Sum(x_ijkt[i, :, :, t].ravel().tolist()))

    # 3. x_ijkt对i和k求和等于spine_num_per_pod
    if not releax:
        for j in range(pod_num):
            for t in range(spine_num_per_pod):
                solver.Add(solver.Sum(x_ijkt[:, j, :, t].ravel().tolist()) == spine_up_port_num)
    else:
        for j in range(pod_num):
            for t in range(spine_num_per_pod):
                solver.Add(spine_up_port_num >= solver.Sum(x_ijkt[:, j, :, t].ravel().tolist()))
                solver.Add(t_jt_dis2[j, t] == spine_up_port_num - solver.Sum(x_ijkt[:, j, :, t].ravel().tolist()))
    for i in range(pod_num):
        for k in range(spine_up_port_num):
            for t in range(spine_num_per_pod):
                solver.Add(solver.Sum(x_ijkt[i, :, k, t].tolist()) <= 1)
    for i in range(pod_num):
        for k in range(spine_up_port_num):
            for t in range(spine_num_per_pod):
                solver.Add(solver.Sum(x_ijkt[:, i, k, t].tolist()) <= 1)
                
    for i in range(pod_num):
        for j in range(pod_num):
            if i == j:
                solver.Add(solver.Sum(x_ijkt[i, j, :, :].ravel().tolist()) == 0)
               
    # 5. h_ijkt >= x_ijkt - u_ijkt
    for i in range(pod_num):
        for j in range(pod_num):
            for k in range(spine_up_port_num):
                for t in range(spine_num_per_pod):
                    solver.Add(h_ijkt[i, j, k, t] >= x_ijkt[i, j, k, t] - u_ijkt[i, j, k, t])
                    solver.Add(h_ijkt[i, j, k, t] >=  u_ijkt[i, j, k, t] - x_ijkt[i, j, k, t])

    # 设置目标函数最小化

    obj = 1*solver.Sum(h_ijkt.ravel().tolist()) + 2.5*solver.Sum(t_ijt_dis.ravel().tolist()) + 1*solver.Sum(t_jt_dis2.ravel().tolist()) + 1*solver.Sum(t_it_dis2.ravel().tolist())
    solver.Minimize(obj)

    # 模型求解
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        np.set_printoptions(threshold=np.inf)
        print(c_ijt)
        print(pod_num, pod_num, spine_up_port_num, spine_num_per_pod)
        return False, None

    # 获取结果
    def get_solution_variable(x):
        return round(float(x.solution_value()))
    get_solution_variable = np.vectorize(get_solution_variable)
    x_ijkt_solution = get_solution_variable(x_ijkt)
    np.set_printoptions(threshold=np.inf)
    # print(x_ijkt_solution)
    return True, x_ijkt_solution


if __name__ == '__main__':
    import sys
    import time
    # open('time.log', 'w').close()



