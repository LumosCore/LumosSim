import math
from ortools.linear_solver import pywraplp
import numpy as np


def solve(C_ij, pod_num, spine_num, spine_up_port_num):
    print("debug pod_num, spine_num, spine_up_port_num",pod_num, spine_num, spine_up_port_num)
    #solver = pywraplp.Solver.CreateSolver('CP_SAT')
    solver = pywraplp.Solver.CreateSolver('Gurobi')
    solver.SetNumThreads(8)
    solver.SetTimeLimit(1000 * 10)
    A_ij = np.empty((pod_num, pod_num), dtype=object)
    A_T_ij = np.empty((pod_num, pod_num), dtype=object)
    for i in range(pod_num):
        for j in range(pod_num):
            A_ij[i, j] = solver.IntVar(0, int(C_ij[i, j]), f'A_ij_{i}_{j}')
            A_T_ij[i, j] = solver.IntVar(0, int(C_ij[i, j]), f'A_T_ij_{i}_{j}')
    obj_val = solver.NumVar(0, 40960, 'obj')

    # 线性化条件
    for i in range(pod_num):
        for j in range(pod_num):
            solver.Add(A_ij[i, j] + A_T_ij[i, j] == C_ij[i, j])
            solver.Add(A_ij[i, j] == A_T_ij[j, i])

    for i in range(pod_num):
        solver.Add(solver.Sum(A_ij[i, :].tolist()) == math.ceil(np.sum(C_ij[i, :]) / 2))
        solver.Add(solver.Sum((A_ij[i, :] + A_T_ij[:, i].T).tolist()) == np.sum(C_ij[i, :]))

    for j in range(pod_num):
        solver.Add(solver.Sum(A_ij[:, j].tolist()) == math.ceil(np.sum(C_ij[:, j]) / 2))
        solver.Add(solver.Sum((A_ij[:, j] + A_T_ij[j, :].T).tolist()) == np.sum(C_ij[:, j]))

    solver.Add(solver.Sum((A_ij + A_T_ij.T).ravel().tolist()) >= obj_val)

    obj = solver.Objective()
    obj.SetCoefficient(obj_val, 1)
    obj.SetMaximization()
    # 开始执行
    status = solver.Solve()
    # 记录运行结果
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        print('The problem does not have an optimal solution.')
        assert False
        return None

    get_solution_variable = np.vectorize(lambda x: x.solution_value())
    A_i_j_solution = get_solution_variable(A_ij)
    A_T_ij_solution = get_solution_variable(A_T_ij)
    if __debug__:
        print("A_i_j: ")
        print(A_i_j_solution)
        print("A_T_ij: ")
        print(A_T_ij_solution)

    return A_i_j_solution, A_T_ij_solution
