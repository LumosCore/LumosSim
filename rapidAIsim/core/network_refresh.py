import copy
from rapidAIsim.core.infrastructure.flow import Flow
import numpy as np
import heapq


def refresh_completion_event():
    """Control global flow_infly_info_dict,
    delete old flow-completion event
    and register earliest-finished flow-completion event into Simulator._event_q.
    The approach can avoid too many events modified in Simulator._event_q.
    """
    from rapidAIsim.core.simulator import Simulator
    from rapidAIsim.core.event.flow_completion_event import FlowCompletionEvent
    switch_port_bandwidth = float(Simulator.CONF_DICT['switch_port_bandwidth'])
    inner_server_bandwidth = float(Simulator.CONF_DICT['inner_server_bandwidth'])
    infra = Simulator.get_infrastructure()
    min_finish_time = float("inf")
    flow_infly_info_dict = infra.get_flow_infly_info_dict()

    current_time = Simulator.get_current_time()
    bandwidth_allocation = Simulator.CONF_DICT['bandwidth_allocation']
    sync_delay_alpha = float(Simulator.CONF_DICT['sync_delay_alpha'])

    if Simulator.CONF_DICT['find_next_hop_method'] == 'rerouting':
        # Try to find flow needing rerouting
        need_reroute_flow_list = []
        for flow_id, flow in flow_infly_info_dict.items():
            if _whether_flow_reroute(flow, infra, switch_port_bandwidth) is True:
                need_reroute_flow_list.append(flow)
        infra = Simulator.get_infrastructure()
        # Try to del flow needing rerouting
        for flow in need_reroute_flow_list:
            infra.del_link_flow(flow.get_flow_id(), flow.get_task_id())

        # Try to reroute
        for flow in need_reroute_flow_list:
            flow.set_hop_list(None)
            flow_id = flow.get_flow_id()
            flow.find_hop_list(need_routing_seed=flow_id, need_rerouting=True)
            src = flow.get_src()
            hop_list = flow.get_hop_list()
            # Start subsequent paths.
            tmp_src = src
            for next_hop in hop_list:
                infra.add_link_flow_occupy(flow_id, tmp_src, next_hop, -2)
                tmp_src = next_hop
            infra.set_flow_infly_info(flow_id, flow, flow.get_task_id())

            Simulator.flow_latest_rerouting_time[flow_id] = current_time

    finished_flows = []
    need_modified_flowids = set()
    # Recalculate infly flow consuming time.
    for flow_id, flow in flow_infly_info_dict.items():

        expected_finish_time = flow.get_expected_finish_time()
        if expected_finish_time == -1:
            expected_finish_time = _calculate_and_modify_finish_time(
                flow, infra, switch_port_bandwidth, inner_server_bandwidth, current_time, bandwidth_allocation)

        # hop_count = len(flow.get_hop_list())
        # sync_delay_alpha = sync_delay_alpha (us) + sync_delay_alpha * hop_count
        # TODO: hop_count * propagation delay (100ns)
        # print("debug sync_delay_alpha ",flow_id,sync_delay_alpha,expected_finish_time)
        # expected_finish_time = expected_finish_time + sync_delay_alpha
        expected_finish_time = expected_finish_time

        if expected_finish_time < min_finish_time - 1e-5:
            min_finish_time = expected_finish_time
            finished_flows = [flow]
            need_modified_flowids = set()
            need_modified_flowids.add(flow_id)
        elif -1e-5 < expected_finish_time - min_finish_time < 1e-5:
            finished_flows.append(flow)
            need_modified_flowids.add(flow_id)

    if min_finish_time == Simulator.LAST_MIN_FINISH_TIME and need_modified_flowids == Simulator.NEED_MODIFIED_FLOWIDS:
        return

    if Simulator.FLOW_COMPLETION_EVENT_RECORD is not None:
        Simulator.FLOW_COMPLETION_EVENT_RECORD.change_to_inactive()

    if finished_flows:
        # new_completion_flow = flow_infly_info_dict[need_modified_flowid]

        # Deal with floating-point accuracy problem.
        min_consuming_time = min_finish_time - Simulator.get_current_time()
        if -1e-6 < min_consuming_time < 1e-6:
            min_consuming_time = 0

        flow_completion_event_obj = FlowCompletionEvent(min_consuming_time, finished_flows)
        Simulator.FLOW_COMPLETION_EVENT_RECORD = flow_completion_event_obj
        Simulator.register_event(flow_completion_event_obj)

        Simulator.LAST_MIN_FINISH_TIME = min_finish_time
        Simulator.NEED_MODIFIED_FLOWIDS = need_modified_flowids


def _whether_flow_reroute(flow: Flow, infra, switch_port_bandwidth):
    src = flow.get_src()
    dst = flow.get_dst()
    flow_id = flow.get_flow_id()
    from rapidAIsim.core.simulator import Simulator
    # Deal with only 1 GPU occupation
    if src == dst:
        return False

    # GPUs in the same server.
    if flow.is_in_the_same_server():
        return False

    # flow has been rerouted
    if Simulator.flow_whether_can_rerouting[flow_id] is False:
        return False
    if (flow_id in Simulator.flow_latest_rerouting_time
            and Simulator.get_current_time() - Simulator.flow_latest_rerouting_time[flow_id] < 1):
        return False

    hop_list = flow.get_hop_list()

    min_available_capacity = _uniform_capacity(src, hop_list, switch_port_bandwidth, flow.get_task_id())

    if min_available_capacity < float(Simulator.CONF_DICT['switch_port_bandwidth']):
        return True

    return False


def _calculate_and_modify_finish_time(flow: Flow, infra, switch_port_bandwidth, inner_server_bandwidth,
                                      current_time, bandwidth_allocation):
    src = flow.get_src()
    dst = flow.get_dst()
    remainder_size = flow.get_remainder_size()

    # Deal with only 1 GPU occupation
    if src == dst:
        # remainder_size -= inner_server_bandwidth * (current_time - flow.get_last_calculated_time())
        remainder_size = 0
        flow.set_remainder_size(remainder_size)
        flow.set_last_calculated_time(current_time)
        # finish_time = remainder_size / inner_server_bandwidth + current_time
        finish_time = current_time
        flow.set_expected_finish_time(finish_time)
        return finish_time

    # GPUs in the same server.
    if flow.is_in_the_same_server():
        remainder_size -= inner_server_bandwidth * (current_time - flow.get_last_calculated_time())
        flow.set_remainder_size(remainder_size)
        flow.set_last_calculated_time(current_time)
        finish_time = remainder_size / inner_server_bandwidth + current_time
        flow.set_expected_finish_time(finish_time)
        return finish_time

    hop_list = flow.get_hop_list()

    if flow.get_min_available_capacity() != -1:
        min_available_capacity = flow.get_min_available_capacity()
        remainder_size -= min_available_capacity * (current_time - flow.get_last_calculated_time())
        flow.set_remainder_size(remainder_size)
    #     assert remainder_size != flow.get_size()

    if flow.whether_need_recalculation() is True:
        # Max-min-fairness
        if bandwidth_allocation == 'max_min_fairness':
            bandwidths_dict = _max_min_fairness(flow.get_task_id())
            min_available_capacity = bandwidths_dict[flow.get_flow_id()]
        else:
            # Uniform capacity
            min_available_capacity = _uniform_capacity(src, hop_list, switch_port_bandwidth, flow.get_task_id())
        flow.set_need_recalculation(False)
    else:
        min_available_capacity = flow.get_min_available_capacity()
    remainder_size = max(0,remainder_size) #TODO考虑时延
    max_consuming_time = remainder_size / min_available_capacity
    finish_time = max_consuming_time + current_time + flow.sync_delay_alpha
    flow.set_last_calculated_time(current_time)
    flow.set_min_available_capacity(min_available_capacity)
    flow.set_expected_finish_time(finish_time)
    return finish_time


def _uniform_capacity(src, hop_list, switch_port_bandwidth, task_id: int):
    from rapidAIsim.core.simulator import Simulator
    # Start subsequent paths.
    tmp_src = src
    available_capacity_list = []
    infra = Simulator.get_infrastructure()
    for next_hop in hop_list:

        # Ongoing path is (tmp_src, next_hop)
        the_link_capacity = infra.get_a_link_capacity(tmp_src, next_hop, task_id)

        # print("debug the_link_capacity")
        # print(the_link_capacity,tmp_src, next_hop)

        if (tmp_src >= int(Simulator.CONF_DICT['NIC_num'])
                and next_hop >= int(Simulator.CONF_DICT['NIC_num'])
                and (Simulator.CONF_DICT['joint_scheduler'] not in ['OCSExpander', 'OCSExpander_superPod','ELEExpander','ELEExpander_SuperPod'])):
            tmp_a = tmp_src
            tmp_b = next_hop
            the_link_flow_occupy_num = int(infra.get_link_num(tmp_src, next_hop, task_id))
            if tmp_src < int(Simulator.CONF_DICT['NIC_num']) + int(Simulator.CONF_DICT['leaf_switch_num']):
                # leaf switch
                tmp_a -= int(Simulator.CONF_DICT['NIC_num'])
            else:
                # spine switch
                tmp_a -= int(Simulator.CONF_DICT['NIC_num'])
                tmp_a -= int(Simulator.CONF_DICT['leaf_switch_num'])
            if next_hop < int(Simulator.CONF_DICT['NIC_num']) + int(Simulator.CONF_DICT['leaf_switch_num']):
                # leaf_switch
                tmp_b -= int(Simulator.CONF_DICT['NIC_num'])
            else:
                # spine switch
                tmp_b -= int(Simulator.CONF_DICT['NIC_num'])
                tmp_b -= int(Simulator.CONF_DICT['leaf_switch_num'])
            if f'{tmp_a}_{tmp_b}' in Simulator.contention_link and Simulator.contention_link[f'{tmp_a}_{tmp_b}'] > 0:

                for tmp_task_id in infra.get_link_flow_occupy_dict():
                    link_flow_occupy_list = infra.get_link_flow_occupy_list(tmp_src, next_hop, tmp_task_id)
                    for tmp_flow_id in link_flow_occupy_list:
                        if tmp_task_id != task_id:
                            tmp_flow = infra.get_flow_from_infly_info_dict(tmp_flow_id)
                            if (tmp_flow.get_start_time() <= Simulator.get_current_time()
                                    <= tmp_flow.get_expected_finish_time()):
                                the_link_flow_occupy_num += 1
        else:
            the_link_flow_occupy_num = int(infra.get_link_num(tmp_src, next_hop, task_id))
        available_capacity = (the_link_capacity / the_link_flow_occupy_num)
        if available_capacity > switch_port_bandwidth:
            available_capacity = switch_port_bandwidth
        leaf_gpu_num = int(Simulator.CONF_DICT['NIC_num']) + int(Simulator.CONF_DICT['leaf_switch_num'])
        if tmp_src >= leaf_gpu_num and next_hop >= leaf_gpu_num:
            with open('./cross_pod_flow.log',mode='a') as f:
                f.write(f"flow contention,{Simulator.get_current_time()}, {task_id},{tmp_src-leaf_gpu_num},{next_hop-leaf_gpu_num},{(tmp_src-leaf_gpu_num)//8},{(next_hop-leaf_gpu_num)//8},{the_link_capacity},{available_capacity},{the_link_flow_occupy_num}\n")

        # tmp debug
        if available_capacity < switch_port_bandwidth:
            # tmp_stage_num = (next_hop - int(Simulator.CONF_DICT['NIC_num'])) // int(
            #     Simulator.CONF_DICT['leaf_switch_num']) + 1
            # print("debug the_link_flow_occupy_num",
            #       tmp_stage_num,available_capacity,tmp_src,next_hop,the_link_flow_occupy_num)

            np.random.seed(int(Simulator.get_current_time()))
            # rand_list = np.random.dirichlet(np.ones(the_link_flow_occupy_num),size=1)
            # print(rand_list,Simulator.get_current_time())
            # cur_conflict_task_list = []
            # for flow_id in infra.get_link_flow_occupy_list(tmp_src, next_hop, taskid):
            #     tmp_flow = infra.get_flow_infly_info_dict()[flow_id]
            #     print("debug flow contention", taskid, tmp_src, next_hop,
            #           the_link_capacity, the_link_flow_occupy_num, tmp_flow._src,tmp_flow._dst,tmp_flow._round_id)
            #     cur_conflict_task_list.append(tmp_flow.get_task_id())
            # print("debug contention", cur_conflict_task_list)
            # cur_in = cur_conflict_task_list.index(taskid)
            # available_capacity = switch_port_bandwidth*rand_list[0][cur_in]
            if ('best' in Simulator.CONF_DICT
                    and Simulator.CONF_DICT['best'] == 'yes'
                    and (Simulator.is_spine_switch(tmp_src) or Simulator.is_spine_switch(next_hop))):
                available_capacity = switch_port_bandwidth
            if ('ignore_intra_pod' in Simulator.CONF_DICT
                    and Simulator.CONF_DICT['ignore_intra_pod'] == 'yes'
                    and ( (Simulator.is_spine_switch(tmp_src) and  Simulator.is_leaf_switch(next_hop))  or (Simulator.is_spine_switch(next_hop) and  Simulator.is_leaf_switch(tmp_src)) )):
                available_capacity = switch_port_bandwidth
            if 'is_two_iter' in Simulator.CONF_DICT and Simulator.CONF_DICT['is_two_iter'] == 'yes':
                if tmp_src >= int(Simulator.CONF_DICT['NIC_num']) and next_hop >= int(Simulator.CONF_DICT['NIC_num']):
                    if (tmp_src
                            < int(Simulator.CONF_DICT['NIC_num']) + int(Simulator.CONF_DICT['leaf_switch_num'])
                            <= next_hop):
                        available_capacity = switch_port_bandwidth
                    elif (tmp_src
                          >= int(Simulator.CONF_DICT['NIC_num']) + int(Simulator.CONF_DICT['leaf_switch_num'])
                          > next_hop):
                        available_capacity = switch_port_bandwidth
            if Simulator.CONF_DICT['joint_scheduler'] in ['OCSExpander', 'OCSExpander_superPod','ELEExpander','ELEExpander_SuperPod']:
                with open('conflict_status.txt', 'a') as f3:
                    link_stage = '{}-{}'.format(Simulator.get_node_type(tmp_src), Simulator.get_node_type(next_hop))
                    conflict_data = [task_id, switch_port_bandwidth / available_capacity, available_capacity, Simulator.get_current_time(),
                                     src, hop_list[-1], tmp_src, next_hop, link_stage,
                                     Simulator.get_node_type(tmp_src), Simulator.get_node_type(next_hop)
                                     ]
                    line = ','.join(map(str, conflict_data))
                    f3.write(line)
                    f3.write("\n")
            # print("warning:check whether conflict", available_capacity, 
            #       infra.get_link_flow_occupy_list(tmp_src, next_hop, taskid),tmp_src,next_hop)
            # for flow_id in infra.get_link_flow_occupy_list(tmp_src, next_hop, taskid):
            #     temp_tmp_src = tmp_src
            #     temp_next_hop = next_hop
            #     if temp_tmp_src>=512 and temp_tmp_src<544:
            #         temp_tmp_src = temp_tmp_src-512
            #         print("start from leaf: ",temp_tmp_src)
            #     if temp_tmp_src>=544:
            #         temp_tmp_src = temp_tmp_src-544
            #         print("start from spine: ",temp_tmp_src)
            #     if 512 <= temp_next_hop < 544:
            #         temp_next_hop = temp_next_hop-512
            #         print("End to leaf: ",temp_next_hop)
            #     if temp_next_hop >= 544:
            #         temp_next_hop = temp_next_hop-544
            #         print("End to spine: ",temp_next_hop)
            #     tmp_flow = infra.get_flow_infly_info_dict()[flow_id]
            #     print(the_link_capacity, tmp_flow._src, tmp_flow._dst, tmp_flow._taskid)
            # print(infra.get_link_flow_occupy_list(tmp_src, next_hop, taskid))
        ###########

        available_capacity_list.append(available_capacity)

        # Update next hop path
        tmp_src = next_hop

    min_available_capacity = min(available_capacity_list)
    return min_available_capacity


def _max_min_fairness(taskid):
    from rapidAIsim.core.simulator import Simulator

    infra = Simulator.get_infrastructure()
    flow_infly_info_dict = infra.get_flow_infly_info_dict()
    bandwidths_dict = {}  # Available bandwidth of every flow after calculation. {flow_id: bandwidth}
    flow_ok_dict = {}  # A flow available bandwidth achieves the upper bound when value is True. {flow_id: True/False}
    ok_flow_cnt = 0  # the number of flows whose calculation is finished.
    num_flows = len(flow_infly_info_dict)

    for flow_id in flow_infly_info_dict.keys():
        bandwidths_dict[flow_id] = 0
        flow_ok_dict[flow_id] = False

    residual_capacities = copy.deepcopy(infra.get_link_capacity_dict(taskid))  # {(src, dst): capacity}
    flow_passed_dict = copy.deepcopy(
        infra.get_link_flow_occupy_dict_given_task_id(taskid))  # {(src, dst): [flow_id, flow_id, ...]}

    while ok_flow_cnt != num_flows:
        # Find feasible bandwidth which could be added.
        min_avg_bandwidth = float('inf')
        avg_bandwidth = float('inf')
        for (src, dst) in residual_capacities.keys():
            if residual_capacities[(src, dst)] < 0.00001:
                continue
            if len(flow_passed_dict[(src, dst)]) > 0:
                avg_bandwidth = residual_capacities[(src, dst)] / len(flow_passed_dict[(src, dst)])
            if avg_bandwidth < min_avg_bandwidth:
                min_avg_bandwidth = avg_bandwidth

        # Add bandwidth for flows
        for flow_id, flow in flow_infly_info_dict.items():
            if flow_ok_dict[flow_id] is False:
                bandwidths_dict[flow_id] += min_avg_bandwidth

        full_links_list = []  # Record links which were fully occupied.
        # Adjust residual link bandwidth.
        for (src, dst) in residual_capacities.keys():
            if residual_capacities[(src, dst)] < 0.00001:  # 0000000000001
                continue

            residual_capacities[(src, dst)] -= min_avg_bandwidth * len(flow_passed_dict[(src, dst)])
            # A link is fully occupied.
            if residual_capacities[(src, dst)] < 0.00001:
                full_links_list.append((src, dst))

        # Remove computed flows on links.
        for (src, dst) in full_links_list:
            for flow_id in flow_passed_dict[(src, dst)]:
                hop_list = flow_infly_info_dict[flow_id].get_hop_list()
                tmp_src = flow_infly_info_dict[flow_id].get_src()
                for next_hop in hop_list:
                    if src == tmp_src and next_hop == dst:
                        continue
                    if (flow_passed_dict.get((tmp_src, next_hop))
                            and flow_id in flow_passed_dict.get((tmp_src, next_hop))):
                        flow_passed_dict[(tmp_src, next_hop)].remove(flow_id)
                    tmp_src = next_hop

                if flow_ok_dict[flow_id] is False:
                    flow_ok_dict[flow_id] = True
                    ok_flow_cnt += 1
            del flow_passed_dict[(src, dst)]
    return bandwidths_dict


def handle_task_finish_immediately(influenced_task_id):
    from rapidAIsim.core.simulator import Simulator
    trigger_time = Simulator.get_current_time()
    from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent
    from rapidAIsim.core.event.flow_completion_event import FlowCompletionEvent

    # 将相关事件失活
    total_increase_comp = 0
    total_increase_comm = 0
    for obj in Simulator._event_q:
        if (obj.event_time >= trigger_time) and obj.is_active:
            if isinstance(obj, FlowTransmitEvent):
                tmp_flow = obj.flow_list[0]
                if tmp_flow.get_task_id() == influenced_task_id:
                    obj.change_to_inactive()
                    Simulator.task_has_computation_time[influenced_task_id] += trigger_time - obj._task_init_time
                    total_increase_comp += trigger_time - obj._task_init_time
            elif isinstance(obj, FlowCompletionEvent):
                for tmp_flow in obj.flows:
                    if tmp_flow.get_task_id() == influenced_task_id:
                        obj.change_to_inactive()
                        break
    heapq.heapify(Simulator._event_q)
    # 在Simulator中清空task_record_dict与wait_transmit_dict
    Simulator._task_record_dict[influenced_task_id] = []
    finish_round_id = Simulator.ITERATION_FINISH_ROUNDID_DICT[influenced_task_id][-1]
    for round_id in range(finish_round_id + 1):
        Simulator._wait_transmit_dict[f'{influenced_task_id}_{round_id}'] = {}
    # print(Simulator._wait_transmit_dict[f'{influenced_task_id}_{0}'])
    # 找到所有正在传输的流
    infra = Simulator.get_infrastructure()
    influenced_flow_list = []
    flow_infly_info_dict = infra._flow_infly_info_dict
    for flow_id, flow in flow_infly_info_dict.items():
        if flow.get_task_id() == influenced_task_id:
            # if influenced_task_id == 28:
            #     print("try to finish flow ",flow_id)
            influenced_flow_list.append(flow)
    # 删除每一条路上的流
    inner_server_bandwidth = float(Simulator.CONF_DICT['inner_server_bandwidth'])
    for flow in influenced_flow_list:
        src = flow.get_src()
        flow_id = flow.get_flow_id()
        taskid = flow.get_task_id()
        hop_list = flow.get_hop_list()
        remainder_size = flow.get_remainder_size()
        if flow.is_in_the_same_server():
            remainder_size -= inner_server_bandwidth * (trigger_time - flow.get_last_calculated_time())
        if flow.get_min_available_capacity() != -1:
            min_available_capacity = flow.get_min_available_capacity()
            remainder_size -= min_available_capacity * (trigger_time - flow.get_last_calculated_time())
        Simulator.task_has_communication_size[taskid] += (flow.get_size() - remainder_size)
        total_increase_comm += (flow.get_size() - remainder_size)
        tmp_src = src
        for next_hop in hop_list:
            infra.del_link_flow_occupy(flow_id, tmp_src, next_hop, taskid)
            tmp_src = next_hop
        infra.del_a_flow_infly_info(flow_id, taskid)
        print(f'finish immediately flow {flow_id} of {taskid}')
    print(
        f'finish immediately taskid, {influenced_task_id}, finish_time, {Simulator.get_current_time()} comp:{total_increase_comp} comm:{total_increase_comm}')
    # 释放资源
    scheduler = Simulator.get_scheduler()
    Simulator.task_time_logger.write(f'taskid,{influenced_task_id},finish_time,{Simulator.get_current_time()}\n')
    scheduler.update_finished_job(influenced_task_id, Simulator.get_current_time(), Simulator.WAITING_TASK_LIST)
    if influenced_task_id in Simulator.need_immediately_finish_task:
        Simulator.need_immediately_finish_task.remove(influenced_task_id)
