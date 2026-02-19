from ortools.linear_solver import pywraplp
import numpy as np


def solve(cluster_pod_num, cluster_spine_num, spine_up_port_num, T_a_b, solver='gurobi'):
    # 输入检查
    pod_num = cluster_pod_num
    if not isinstance(T_a_b, np.ndarray):
        T_a_b = np.array(T_a_b)
    assert T_a_b.shape == (pod_num, pod_num)
    # 验证T_a_b是否为对称矩阵
    if not np.allclose(T_a_b, T_a_b.T):
        raise ValueError('T_a_b不是对称矩阵')
    T_a_b_copy = T_a_b.copy().astype(np.float32)
    # “归一化”T_a_b变量，使得T_a_b每一行的和与cluster_spine_num * spine_up_port_num接近，这样可以使得计算方差时
    # 不受到T_a_b本身数值大小的影响，而使得最终结果在数值的比例上与T_a_b保持一致
    if np.any(T_a_b_copy.ravel()):
        line_sum = T_a_b_copy.sum(axis=0)
        k = cluster_spine_num * spine_up_port_num * np.sum(line_sum) / np.sum(line_sum ** 2)
        T_a_b_copy *= k

    # 建模
    if solver == 'cp_sat':
        solver = pywraplp.Solver.CreateSolver('CP_SAT')
        solver.SetNumThreads(14)
        solver.SetTimeLimit(1000 * 1000)
        # solver.SetSolverSpecificParametersAsString("OutputFlag=1")
    elif solver == 'scip':
        solver = pywraplp.Solver.CreateSolver('SCIP')
        solver.SetNumThreads(14)
        solver.SetTimeLimit(1000 * 1000)
    elif solver == 'gurobi':
        solver = pywraplp.Solver.CreateSolver('GUROBI')
        # solver.SetSolverSpecificParametersAsString("MIPGap=0.2")
        # solver.SetSolverSpecificParametersAsString("OutputFlag=1")
        solver.SetTimeLimit(1000 * 1000)
    else:
        raise ValueError('solver参数错误')

    # 设置模型变量
    t_a_b = np.empty((pod_num, pod_num), dtype=object)
    for a in range(pod_num):
        for b in range(pod_num):
            if a != b:
                t_a_b[a, b] = solver.IntVar(int(T_a_b_copy[a, b] > 0), cluster_spine_num * spine_up_port_num,
                                            f't_a_b_{a}_{b}')
            else:
                t_a_b[a, b] = solver.IntVar(0, 0, f't_a_b_{a}_{b}')
    abs_delta_a_b = np.empty((pod_num, pod_num), dtype=object)
    for a in range(pod_num):
        for b in range(pod_num):
            if a != b:
                abs_delta_a_b[a, b] = solver.IntVar(0, cluster_spine_num * spine_up_port_num, f'abs_delta_a_b_{a}_{b}')
            else:
                abs_delta_a_b[a, b] = solver.IntVar(0, 0, f'abs_delta_a_b_{a}_{b}')
    # sum_r_a_b = solver.IntVar(0, pod_num * pod_num * spine_up_port_num, 'sum_r_a_b')
    avg_t_a_b = solver.NumVar(0, cluster_spine_num * spine_up_port_num, 'avg_t_a_b')
    sum_abs = solver.IntVar(0, cluster_spine_num * spine_up_port_num * pod_num * pod_num * pod_num, 'sum_abs')
    l1norm_r_a_b = np.empty((pod_num, pod_num), dtype=object)
    for a in range(pod_num):
        for b in range(pod_num):
            if a != b:
                l1norm_r_a_b[a, b] = solver.NumVar(0, cluster_spine_num * spine_up_port_num, f'l1norm_r_a_b_{a}_{b}')
            else:
                l1norm_r_a_b[a, b] = solver.NumVar(0, 0, f'l1norm_r_a_b_{a}_{b}')
    sum_l1norm = solver.NumVar(0, cluster_spine_num * spine_up_port_num * pod_num * pod_num * pod_num, 'sum_l1norm')

    # t_a_b = t_b_a
    for a in range(pod_num):
        for b in range(pod_num):
            if a != b:
                solver.Add(t_a_b[a, b] == t_a_b[b, a])
    # 求和约束
    for a in range(pod_num):
        solver.Add(solver.Sum(t_a_b[a, :].tolist()) == cluster_spine_num * spine_up_port_num)
        solver.Add(solver.Sum(t_a_b[:, a].tolist()) == cluster_spine_num * spine_up_port_num)
    # 绝对值约束
    for a in range(pod_num):
        for b in range(pod_num):
            if a != b:
                solver.Add(abs_delta_a_b[a, b] >= (t_a_b[a, b] - float(T_a_b_copy[a, b])))
                solver.Add(abs_delta_a_b[a, b] >= (float(T_a_b_copy[a, b]) - t_a_b[a, b]))
    # 平均值约束
    solver.Add(avg_t_a_b == solver.Sum(t_a_b.ravel().tolist()) / ((pod_num - 1) * pod_num))
    # solver.Add(sum_r_a_b == solver.Sum(t_a_b.ravel().tolist()))
    solver.Add(sum_abs == solver.Sum(abs_delta_a_b.ravel().tolist()))
    # L1范数约束
    for a in range(pod_num):
        for b in range(pod_num):
            if a != b:
                solver.Add(l1norm_r_a_b[a, b] >= t_a_b[a, b] - avg_t_a_b)
                solver.Add(l1norm_r_a_b[a, b] >= avg_t_a_b - t_a_b[a, b])
    solver.Add(sum_l1norm == solver.Sum(l1norm_r_a_b.ravel().tolist()))

    # 目标函数
    # sum_t_a_b_obj = solver.Objective()
    # sum_t_a_b_obj.SetCoefficient(sum_r_a_b, 1)
    # sum_t_a_b_obj.SetMaximization()

    min_abs_obj = solver.Objective()
    min_abs_obj.SetCoefficient(sum_abs, 100)
    min_abs_obj.SetMinimization()

    min_l1norm_obj = solver.Objective()
    min_l1norm_obj.SetCoefficient(sum_l1norm, 1)
    min_l1norm_obj.SetMinimization()
    # solver_params = pywraplp.MPSolverParameters()
    # solver_params.SetDoubleParam(solver_params.RELATIVE_MIP_GAP, 0.3)
    # start_t = time.time()
    status = solver.Solve()
    # end_t = time.time()
    # solve_time = end_t - start_t
    # print(solve_time)
    # num_vars = solver.NumVariables()
    # print('Number of variables =', num_vars)

    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        # print('The problem does not have an optimal solution.')
        return None

    # status = 'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'

    # print('Optimal objective: %g' % solver.Objective().Value())

    def get_solution_variable(x):
        return x.solution_value()
    get_solution_variable = np.vectorize(get_solution_variable)
    t_a_b_solution = get_solution_variable(t_a_b)
    # var = {'o_i_a_b_k': o_i_a_b_k_values}
    return t_a_b_solution.astype(np.int32)
