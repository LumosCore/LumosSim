import numpy as np

from routing_solver import RoutingSolver


def test_routing_solver():
    NIC_num = 2048
    spine_switch_num = 256
    spine_switch_port_num = 16
    pod_num = 32
    server_num_per_pod = 8
    routing_solver = RoutingSolver(pod_num, spine_switch_num, NIC_num, server_num_per_pod,
                                   spine_switch_port_num // 2, 3.0)
    x_ijkt = np.load('x_ijkt.npy')
    print(np.sum(x_ijkt, axis=(2, 3)))
    routing_solver.generate_routing_table(x_ijkt)
    intra_pod_up_table = routing_solver.get_intra_pod_up_table()
    intra_pod_down_table = routing_solver.get_intra_pod_down_table()
    inter_pod_table = routing_solver.get_inter_pod_routing_table()
    inter_pod_weighted_direct_table = routing_solver.get_inter_pod_weighted_direct_routing_table()
    inter_pod_weighted_twohop_table = routing_solver.get_inter_pod_weighted_twohop_routing_table()
    print(1)


if __name__ == '__main__':
    test_routing_solver()
