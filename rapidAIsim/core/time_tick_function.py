import random
from rapidAIsim.core.infrastructure.flow import Flow

def detect_and_update_flow_every_tick(time_tick):
    from rapidAIsim.core.simulator import Simulator
    from rapidAIsim.core.event.flow_completion_event import FlowCompletionEvent
    switch_port_bandwidth = float(Simulator.CONF_DICT['switch_port_bandwidth'])
    inner_server_bandwidth = float(Simulator.CONF_DICT['inner_server_bandwidth'])
    TCP_type = Simulator.CONF_DICT['TCP_type']
    infra = Simulator.get_infrastructure()
    current_time = Simulator._current_time

    flow_infly_info_dict = infra.get_flow_infly_info_dict()
    
    if TCP_type == 'DCTCP':
        link_occupy_bandwidth_ratio_dict = _mark_CE_on_DCTCP(infra, switch_port_bandwidth)

    # Recalculate infly flow consuming time.
    finish_flow_list = []
    for flowid, flow in flow_infly_info_dict.items():
        if TCP_type == 'DCTCP':
            remainder_size = _modify_remainder_size_on_DCTCP(flow, time_tick, infra, current_time, switch_port_bandwidth, inner_server_bandwidth, link_occupy_bandwidth_ratio_dict)
        elif TCP_type == 'learning':
            remainder_size = _learning_from_netbench_cwnd_log(flow, time_tick, infra, current_time, switch_port_bandwidth, inner_server_bandwidth, Simulator.CWND_RATIO_DICT)
        else:
            remainder_size = _caculate_and_modify_remainder_size(flow, time_tick, infra, current_time, switch_port_bandwidth, inner_server_bandwidth)

        if remainder_size <= 0.0001:
            finish_flow_list.append(flow)

    for finish_flow in finish_flow_list:
        FlowCompletionEvent(0, finish_flow).do_sth()


def _modify_remainder_size_on_DCTCP(flow: Flow, time_tick, infra, current_time, switch_port_bandwidth, inner_server_bandwidth, link_occupy_bandwidth_ratio_dict):
    src = flow.get_src()
    dst = flow.get_dst()
    remainder_size = flow.get_remainder_size()

    # Deal with only 1 GPU occupation or GPUs in the same server.
    if src == dst or flow.is_in_the_same_server():
        remainder_size -= inner_server_bandwidth * time_tick
        flow.set_remainder_size(remainder_size)
        flow.set_last_calculated_time(current_time)
        return remainder_size

    hop_list = flow.get_hop_list()
    
    g = 0.0625
    if flow.get_min_available_capacity() != -1:
        remainder_size -= flow.get_min_available_capacity() * time_tick
        flow.set_remainder_size(remainder_size)
        CEmarked_fraction_F = _find_CEmarked_fraction(src, hop_list, link_occupy_bandwidth_ratio_dict)
        if CEmarked_fraction_F == 0:
            flow._cwnd = switch_port_bandwidth
        else:
            flow._alpha = (1 - g) * flow._alpha + g * CEmarked_fraction_F
            flow._cwnd = flow._cwnd * (1 - flow._alpha / 2)
    else:
        flow._cwnd = switch_port_bandwidth

    flow.set_min_available_capacity(flow._cwnd)
    flow.set_last_calculated_time(current_time)
    return remainder_size


def _find_CEmarked_fraction(src, hop_list, link_occupy_bandwidth_ratio_dict):
    # TODO: Be relative with Netbench.
    # Calculate expected completion time by _cwnd and compare with Netbench.
    
    CEmarked_fraction_F = 0
    next_hop = None
    tmp_src = src
    for next_hop in hop_list:
        if link_occupy_bandwidth_ratio_dict[(tmp_src, next_hop)] > 1:
            tmp_F = 1 - 1 / link_occupy_bandwidth_ratio_dict[(tmp_src, next_hop)]
            if tmp_F > CEmarked_fraction_F:
                CEmarked_fraction_F = tmp_F
        # Update next hop path
        tmp_src = next_hop

    # if CEmarked_fraction_F > 0:
    #     rdm = random.random()
    #     CEmarked_fraction_F += rdm
    #     if CEmarked_fraction_F > 1:
    #         CEmarked_fraction_F = 1
    return CEmarked_fraction_F


def _mark_CE_on_DCTCP(infra, switch_port_bandwidth):
    link_flow_occupy_dict = infra.get_link_flow_occupy_dict_given_task_id(-2)
    flow_infly_info_dict = infra.get_flow_infly_info_dict()
    link_occupy_bandwidth_ratio_dict = {}
    for (src, dst), flow_list in link_flow_occupy_dict.items():
        consume_bandwidth = 0
        for contend_flow_id in flow_list:
            consume_bandwidth += flow_infly_info_dict[contend_flow_id]._cwnd
        link_occupy_bandwidth_ratio_dict[(src, dst)] = consume_bandwidth / switch_port_bandwidth
    return link_occupy_bandwidth_ratio_dict


def _caculate_and_modify_remainder_size(flow: Flow, time_tick, infra, current_time, switch_port_bandwidth, inner_server_bandwidth):
    src = flow.get_src()
    dst = flow.get_dst()
    remainder_size = flow.get_remainder_size()

    # Deal with only 1 GPU occupation or GPUs in the same server.
    if src == dst or flow.is_in_the_same_server():
        remainder_size -= inner_server_bandwidth * time_tick
        flow.set_remainder_size(remainder_size)
        flow.set_last_calculated_time(current_time)
        return remainder_size

    hop_list = flow.get_hop_list()

    if flow.get_min_available_capacity() != -1:
        remainder_size -= flow.get_min_available_capacity() * time_tick
        flow.set_remainder_size(remainder_size)

    if flow.whether_need_recalculation() == True:
        # Max-min-fairness
        # if Simulator.CONF_DICT['bandwidth_allocation'] == 'max_min_fairness':
        #     bandwidths_dict = _max_min_fairness()
        #     min_available_capacity = bandwidths_dict[flow.get_flow_id()]
        # else:
        # Uniform capacity
        # TODO: Add variates
        min_available_capacity = _uniform_capacity(src, hop_list, infra, switch_port_bandwidth, flow.get_task_id())
        flow.set_need_recalculation(False)
    else:
        min_available_capacity = flow.get_min_available_capacity()
    flow.set_min_available_capacity(min_available_capacity)
    flow.set_last_calculated_time(current_time)
    return remainder_size


def _uniform_capacity(src, hop_list, infra, switch_port_bandwidth, taskid):
    # Start subsequent paths.
    next_hop = None
    tmp_src = src
    available_capacity_list = []
    for next_hop in hop_list:
        # Ongoing path is (tmp_src, next_hop)
        the_link_capacity = infra.get_a_link_capacity(tmp_src, next_hop, taskid)

        # Uniform capacity
        the_link_flow_occupy_num = len(infra.get_link_flow_occupy_list(tmp_src, next_hop, taskid))
        available_capacity = (the_link_capacity / the_link_flow_occupy_num) if the_link_flow_occupy_num > 0 else the_link_capacity

        if available_capacity > switch_port_bandwidth:
            available_capacity = switch_port_bandwidth
        
        # tmp debug
        # if available_capacity < switch_port_bandwidth:
        #     print("warning:check whether conflict")
        ###########

        available_capacity_list.append(available_capacity)

        # Update next hop path
        tmp_src = next_hop

    min_available_capacity = min(available_capacity_list)
    return min_available_capacity


def _learning_from_netbench_cwnd_log(flow: Flow, time_tick, infra, current_time, switch_port_bandwidth, inner_server_bandwidth, cwnd_ratio_dict):
    src = flow.get_src()
    dst = flow.get_dst()
    remainder_size = flow.get_remainder_size()

    # Deal with only 1 GPU occupation or GPUs in the same server.
    if src == dst or flow.is_in_the_same_server():
        remainder_size -= inner_server_bandwidth * time_tick
        flow.set_remainder_size(remainder_size)
        flow.set_last_calculated_time(current_time)
        return remainder_size

    flowid = flow.get_flow_id()
    flow_infly_info_dict = infra.get_flow_infly_info_dict()
    infly_flowid_tuple = tuple(flow_infly_info_dict.keys())
    
    if len(infly_flowid_tuple) > 1:
        cwnd = cwnd_ratio_dict[infly_flowid_tuple][flowid] * switch_port_bandwidth
    else:
        cwnd = switch_port_bandwidth

    remainder_size -= cwnd * time_tick
    flow.set_remainder_size(remainder_size)
    flow.set_last_calculated_time(current_time)
    return remainder_size
