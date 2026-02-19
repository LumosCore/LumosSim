import numpy as np


def get_traffic_matrix(curr_time) -> np.ndarray:
    """
    获得当前时刻的流量矩阵。请注意，调用该函数后，会将流量矩阵清空！
    """
    from rapidAIsim.core.simulator import Simulator
    infra = Simulator.get_infrastructure()
    flow_infly_info = infra.get_flow_infly_info_dict()
    traffic_matrix = Simulator.get_traffic_matrix()
    print(traffic_matrix)
    for flow in flow_infly_info.values():
        hop_list = flow.get_interAS_hop_list()
        if not hop_list:  # intra-server, intra-leaf / intra-pod flow
            continue
        # calculate the transmit size
        flow.last_statistic_time = curr_time
        remainder_size = flow.get_remainder_size()
        min_available_capacity = flow.get_min_available_capacity()
        remainder_size -= min_available_capacity * (curr_time - flow.get_last_calculated_time())
        transmit_size = flow.traffic_matrix_remainder_size - remainder_size
        flow.traffic_matrix_remainder_size = remainder_size
        if transmit_size < 0:
            raise ValueError(f'Invalid transmit size: {transmit_size}. Details: {flow}')

        # update traffic matrix
        for src, dst, index in hop_list:
            traffic_matrix[src, dst, index] += transmit_size
    return_val = np.copy(traffic_matrix)
    Simulator.reset_traffic_matrix()
    return return_val
