"""
该模块用于将对称的链路需求矩阵分解为两个互为转置的矩阵之和。
"""


import math
from ortools.linear_solver import pywraplp
import numpy as np


def solve(C_ij, pod_num):
    solver = pywraplp.Solver.CreateSolver('gurobi')
    solver.SetNumThreads(8)
    #solver.SetTimeLimit(1000 * 10)
    A_ij = np.empty((pod_num, pod_num), dtype=object)
    A_T_ij = np.empty((pod_num, pod_num), dtype=object)
    
    A_ij_res = np.empty((pod_num, pod_num), dtype=int)
    A_T_ij_res = np.empty((pod_num, pod_num), dtype=int)
    
    if np.all(C_ij % 2 == 0):
        A_ij_res = C_ij // 2
        A_T_ij_res = C_ij // 2
        # print("find jiejing")
        return A_ij_res,A_T_ij_res
    
    
    
    for i in range(pod_num):
        for j in range(pod_num):
            A_ij[i, j] = solver.IntVar(0, int(C_ij[i, j]), f'A_ij_{i}_{j}')
            A_T_ij[i, j] = solver.IntVar(0, int(C_ij[i, j]), f'A_T_ij_{i}_{j}')
    # obj_val = solver.NumVar(0, 40960, 'obj')

    # 线性化条件
    for i in range(pod_num):
        for j in range(pod_num):
            solver.Add(A_ij[i, j] + A_T_ij[i, j] == C_ij[i, j])
            solver.Add(A_ij[i, j] == A_T_ij[j, i])

    for i in range(pod_num):
        solver.Add(solver.Sum(A_ij[i, :].tolist()) <= math.ceil(np.sum(C_ij[i, :]) / 2))
        solver.Add(solver.Sum(A_ij[i, :].tolist()) >= math.floor(np.sum(C_ij[i, :]) / 2))
        # solver.Add(solver.Sum((A_ij[i, :] + A_T_ij[:, i].T).tolist()) == np.sum(C_ij[i, :]))

    for j in range(pod_num):
        solver.Add(solver.Sum(A_ij[:, j].tolist()) <= math.ceil(np.sum(C_ij[:, j]) / 2))
        solver.Add(solver.Sum(A_ij[:, j].tolist()) >= math.floor(np.sum(C_ij[:, j]) / 2))
        # solver.Add(solver.Sum((A_ij[:, j] + A_T_ij[j, :].T).tolist()) == np.sum(C_ij[:, j]))

    # solver.Add(solver.Sum((A_ij + A_T_ij.T).ravel().tolist()) >= obj_val)

    # obj = solver.Objective()
    # obj.SetCoefficient(obj_val, 1)
    # obj.SetMaximization()
    # 开始执行
    status = solver.Solve()
    # 记录运行结果
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        np.set_printoptions(threshold=np.inf)
        print(C_ij)
        print(pod_num)
        np.save('error_C_ij.npy', C_ij)
        raise RuntimeError('The problem does not have solution.')

    get_solution_variable = np.vectorize(lambda x: round(x.solution_value()))
    A_i_j_solution = get_solution_variable(A_ij)
    A_T_ij_solution = get_solution_variable(A_T_ij)
    # if __debug__:
    #     print("A_i_j: ")
    #     print(A_i_j_solution)
    #     print("A_T_ij: ")
    #     print(A_T_ij_solution)

    return A_i_j_solution, A_T_ij_solution


if __name__ == '__main__':

    
    # [[0,4,7,3],
    #         [4,0,2,2],
    #         [7,2,0,3],
    #         [3,2,3,0],
    #         ]
    C_ij = [[0,4,4,4],
            [4,0,4,4],
            [4,4,0,4],
            [4,4,4,0],
            ]
    C_ij = np.array(C_ij)
    A_i_j_solution, A_T_ij_solution = solve(C_ij,4)
    print(np.array(C_ij))
    print(A_i_j_solution)
    print(A_T_ij_solution)
