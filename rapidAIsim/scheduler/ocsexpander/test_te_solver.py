import TE_solver_lp


def test_TE():
    pod_num = 16
    cluster_spine_num = 16
    spine_up_port_num = 16
    T_a_b = [[3 for _ in range(pod_num)] for _ in range(pod_num)]
    c_ij = TE_solver_lp.solve(pod_num, cluster_spine_num, spine_up_port_num, T_a_b)
    print(c_ij)


if __name__ == '__main__':
    test_TE()
