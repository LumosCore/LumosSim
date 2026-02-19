import time
import warnings


def del_global_record_trigger_new_round(task_id, round_id):
    """After every flow is arrived, check the task and the round if is finished.
    And trigger new round.
    """

    from rapidAIsim.core.simulator import Simulator
    from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

    if _detect_if_a_round_done(task_id, round_id) is True:
        # print(f"Round {round_id} in task {task_id} is done at time {Simulator.get_current_time()}!", flush = True)
        # Trigger next round.
        # print("debug trigger finish",roundid,  flow_id, Simulator.get_current_time())

        if Simulator.get_wait_transmit_dict().get(f'{task_id}_{round_id + 1}'):
            flow_list = []
            for flow_id, flow in Simulator.get_wait_transmit_dict()[f'{task_id}_{round_id + 1}'].items():
                if flow_id not in Simulator.flow_has_started or not Simulator.flow_has_started[flow_id]:
                    # Note that Time recorded in Flow structure is absolute time,
                    # while time in Event triggered by Simulator is relative time. 
                    flow.set_last_calculated_time(Simulator.get_current_time())
                    flow_list.append(flow)
                    Simulator.GPU_status[flow.get_src()] = 1
                    Simulator.GPU_status[flow.get_dst()] = 1
                    # print("debug1 flow start",  flow_id, Simulator.get_current_time(), roundid)
                    Simulator.flow_has_started[flow_id] = True
            round_id_flag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[task_id]
            iteration_round_num = round_id_flag_list[0] + 1
            computation_time = Simulator.TASK_LIST[task_id].computation_time
            assert computation_time > 0, f"The computation time is not set. Task id: {task_id}"
            if Simulator.CONF_DICT['task_type'] != 'llm':
                # If a communication iteration is finished, add computation time delay.
                if (round_id + 1) % iteration_round_num == 0:
                    print(f"Iteration {(round_id + 1) // iteration_round_num} in task {task_id} at round {round_id} "
                          f"finished at time {Simulator.get_current_time()}.")
                    Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))
                else:
                    Simulator.register_event(FlowTransmitEvent(0, flow_list))
            else:
                Simulator.subsequent_time = 0
                Simulator.prior_time = 0
                if (round_id + 1) % iteration_round_num == 0:
                    print(f"Iteration {(round_id + 1) // iteration_round_num} in task {task_id} at round {round_id} "
                          f"finished at time {Simulator.get_current_time()}.")
                else:
                    for flow in flow_list:
                        if flow.need_subsequent_computation:
                            Simulator.subsequent_time = computation_time
                            break
                for flow in flow_list:
                    if flow.need_prior_computation:
                        Simulator.prior_time = computation_time
                        break
                computation_time = Simulator.subsequent_time + Simulator.prior_time
                # print(f'debug round {round_id} flow {flow_list[0].get_flow_id()} of task {taskid}
                # with computation {computation_time} ')
                sync_delay_alpha = 0
                if Simulator.CONF_DICT['joint_scheduler'] in ['OCSExpander_superPod','OCSExpander']:
                    sync_delay_alpha = 3*float(Simulator.CONF_DICT['sync_delay_alpha'])
                elif Simulator.CONF_DICT['joint_scheduler'] in ['ELEExpander','ELEExpander_SuperPod']:
                    sync_delay_alpha = 4*float(Simulator.CONF_DICT['sync_delay_alpha'])
                # for flow in flow_list:
                #     print("debug sync_delay_alpha",sync_delay_alpha, computation_time,flow._flow_id, flow._src,  flow._dst, flow._size)
                Simulator.register_event(FlowTransmitEvent(computation_time+sync_delay_alpha, flow_list))

    if Simulator.is_task_done(task_id):
        iteration_round_num = Simulator.ITERATION_FINISH_ROUNDID_DICT[task_id][0] + 1
        print(f"Iteration {(round_id + 1) // iteration_round_num} in task {task_id} at round {round_id} "
              f"finished at time {Simulator.get_current_time()}.")
        exp_comm_time = Simulator.task_expected_comm_time[task_id]
        actual_comm_time = Simulator.task_actual_comm_time[task_id]
        # exp_comp_time = Simulator.task_need_comp_time[task_id]
        # actual_comp_time = Simulator.task_has_computation_time[task_id]
        print(f'Task {task_id} is done at time {Simulator.get_current_time()}!\n'
              f'Expected communication time: {exp_comm_time}\n'
              f'Actual communication time: {actual_comm_time}')
              # f'Expected computation time: {exp_comp_time}\n'
              # f'Actual computation time: {actual_comp_time}')
        if exp_comm_time - actual_comm_time > 1e-3:
            warn_text = (f'Task {task_id} expected communication time is larger than actual communication time!\n'
                         f'Task info: {Simulator.TASK_LIST[task_id]}')
            warnings.warn(warn_text)
            ratio = 1.0
        else:
            ratio = actual_comm_time / exp_comm_time
        Simulator.task_comm_ratio_logger.write(f'{task_id},{exp_comm_time},{actual_comm_time},{ratio}\n')
        Simulator.task_time_logger.write(f'taskid,{task_id},finish_time,{Simulator.get_current_time()}\n')

        scheduler = Simulator.get_scheduler()
        scheduler.update_finished_job(task_id, Simulator.get_current_time(), Simulator.WAITING_TASK_LIST)
        # temp_gpu_indexes = Simulator._infra_base._job_gpu_list_map[taskid]
        # for temp_gpu_index in temp_gpu_indexes:
        #     assert Simulator._infra_base._used_gpu_state_map[temp_gpu_index] == 1
        #     Simulator._infra_base._used_gpu_state_map[temp_gpu_index] = 0
        if task_id in Simulator.need_immediately_finish_task:
            Simulator.need_immediately_finish_task.remove(task_id)
        _detect_and_trigger_a_task()


def _detect_if_a_round_done(task_id, round_id):
    """Detect whether a round is completed when a flow is completed,
    """
    from rapidAIsim.core.simulator import Simulator
    wait_transmit_dict = Simulator.get_wait_transmit_dict()
    if wait_transmit_dict[f'{task_id}_{round_id}'] == {}:
        return True
    else:
        return False


def continue_record_more_iteration_if_need(task_id, task_occupied_NIC_num, model_size, task_type_obj, iteration_num,
                                           NIC_num_in_a_server, use_NIC_list):
    from rapidAIsim.core.simulator import Simulator

    round_pair_list = task_type_obj.get_task_a_iteration_pair_list(
        task_occupied_NIC_num, model_size, NIC_num_in_a_server)
    iteration_round_num = len(round_pair_list)
    Simulator.ITERATION_FINISH_ROUNDID_DICT[task_id] = [iteration_round_num * i - 1 for i in
                                                        range(1, iteration_num + 1)]

    # 从第二个iteration开始，生成flow对象，并加入等待传输队列。
    from rapidAIsim.core.infrastructure.flow import Flow
    round_id = len(round_pair_list)
    for _ in range(1, iteration_num):
        for pair_list in round_pair_list:
            # Every round
            # print("debug pair_list",pair_list)
            if Simulator.CONF_DICT['joint_scheduler'] in ['OCSExpander', 'OCSExpander_superPod', 'ELEExpander','ELEExpander_SuperPod']:
                for src, dst, communication_size, need_prior_calculate, need_subsequent_calculate in pair_list:
                    flow = Flow(Simulator.FLOWID, communication_size, None, int(src), int(dst), communication_size,
                                None, task_id, round_id, task_occupied_NIC_num, False,
                                need_prior_calculate=need_prior_calculate,
                                need_subsequent_calculate=need_subsequent_calculate)
                    task_type_obj.record_network_occupy(task_id, round_id, flow, src)
                    Simulator.FLOWID += 1
            else:
                for src, dst, communication_size in pair_list:
                    flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                                communication_size, None, task_id, round_id, task_occupied_NIC_num, False)
                    task_type_obj.record_network_occupy(task_id, round_id, flow, use_NIC_list[src])
                    Simulator.FLOWID += 1
            round_id += 1


def _detect_and_trigger_a_task():
    from rapidAIsim.core.simulator import Simulator

    # Modify according to scheduling algorithm.
    if len(Simulator.WAITING_TASK_LIST) > 0:
        scheduler = Simulator.get_scheduler()
        while True:
            a_waiting_task = Simulator.pop_a_waiting_task()
            arriving_time, model_size, task_occupied_NIC_num, task_type_obj, task_id = a_waiting_task.get_task_info()
            allocate_succeed, use_NIC_list = allocate_a_task(scheduler, model_size, task_occupied_NIC_num,
                                                             task_type_obj, task_id)
            if allocate_succeed is True:
                NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
                task_iteration_num = int(Simulator.CONF_DICT['task_iteration_num'])
                continue_record_more_iteration_if_need(task_id, task_occupied_NIC_num, model_size, task_type_obj,
                                                       task_iteration_num, NIC_num_in_a_server, use_NIC_list)
                if len(Simulator.WAITING_TASK_LIST) == 0:
                    return
            else:
                # print('tmp debug', Simulator.get_current_time(), task_id, model_size, task_occupied_NIC_num)
                Simulator.push_a_waiting_task(a_waiting_task)
                break
        return


def allocate_a_task(scheduler, model_size, task_occupied_NIC_num, task_type_obj, task_id):
    """
    start event触发时调用、需要调度新任务是也会调用这个函数。
    """
    from rapidAIsim.core.simulator import Simulator
    NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
    gpu_indexes = []
    conservative = Simulator.CONF_DICT['find_next_hop_method'] == 'conservative'

    not_need_refresh = True
    special_pair = None
    contention_link = None
    joint_scheduler = Simulator.CONF_DICT['joint_scheduler']
    base_time = time.time()
    # static means that do not reconfigure topology.
    # static_scheduler means scheduler based on switches subClos.
    if joint_scheduler in ['static', 'static_scheduler', 'static_scheduler_small',
                           'NaiveScheduler', 'NaiveSchedulerCompare']:
        greedy = Simulator.CONF_DICT['greedy'] == 'yes'
        not_greedy = not greedy
        allocate_succeed, gpu_indexes, allocated_link_mapping, special_pair, _ = scheduler.schedule(
            task_occupied_NIC_num, task_id, Simulator.get_current_time(), Simulator.WAITING_TASK_LIST, not_greedy)
    elif joint_scheduler in ['GPUPlacementer', 'GPUPlacementer2', 'StaticPlacementer',
                             'StaticPlacementerRelax', 'StaticPlacementerAI']:
        (allocate_succeed, not_need_refresh, gpu_indexes, allocated_link_mapping, contention_link,
         _, __) = scheduler.schedule(task_occupied_NIC_num, task_id, Simulator.get_current_time(),
                                     Simulator.WAITING_TASK_LIST, )
    elif joint_scheduler in ['GPUPlacementer3', 'GPUPlacementer4']:
        (allocate_succeed, not_need_refresh, gpu_indexes, allocated_link_mapping, special_pair,
         _, __) = scheduler.schedule(task_occupied_NIC_num, task_id, Simulator.get_current_time(),
                                     Simulator.WAITING_TASK_LIST, "spine_first")
    elif joint_scheduler in ['ELEExpander','OCSExpander_superPod', 'OCSExpander','ELEExpander_SuperPod']:
        task = Simulator.TASK_LIST[task_id]
        TP, PP, DP, EP = task.TP, task.PP, task.DP, task.EP
        allocate_succeed, ep_qp, pp_qp, dp_qp, allocated_link_mapping = scheduler.schedule(TP, DP, PP, EP, task_id, model_size=model_size)
    else:
        allocate_succeed, gpu_indexes, allocated_link_mapping, all_gpu_index, _ = scheduler.schedule(
            task_occupied_NIC_num, task_id, Simulator.get_current_time(), Simulator.WAITING_TASK_LIST, )

    if not allocate_succeed:
        print("Task {} schedule failed!".format(task_id))
        return False, None
    print("Task {} schedule cost: {}".format(task_id, time.time() - base_time))
    # temp_used_leaf_list = []
    # for temp_gpu in gpu_indexes:
    #     temp_leaf_index = int(temp_gpu/32)
    #     if temp_leaf_index not in temp_used_leaf_list:
    #         temp_used_leaf_list.append(temp_leaf_index)

    # # for key in Simulator._infra_base._used_gpu_state_map:
    # server_port_status_list = []
    # for server_id in range(256):
    #     temp_leaf_index = int(server_id/4)
    #     if temp_leaf_index in temp_used_leaf_list:
    #         temp_free_num = 0
    #         for gpu_index in range(server_id*8,server_id*8+8,1):
    #             if gpu_index in Simulator._infra_base._used_gpu_state_map:
    #                 temp_free_num += (1-Simulator._infra_base._used_gpu_state_map[gpu_index])
    #             else:
    #                 temp_free_num += 1
    #         server_port_status_list.append(str(server_id)+": "+str(temp_free_num))
    # # Joint scheduler mode
    # # oxc_scheduler means scheduler based on combining OXC and switches.
    # for temp_gpu_index in gpu_indexes:
    #     assert temp_gpu_index not in Simulator._infra_base._used_gpu_state_map or \
    #         Simulator._infra_base._used_gpu_state_map[temp_gpu_index] == 0
    #     Simulator._infra_base._used_gpu_state_map[temp_gpu_index] = 1
    # Simulator._infra_base._job_gpu_list_map[task_id] = gpu_indexes

    if contention_link is not None and len(contention_link) > 0:
        for k, v in contention_link.items():
            if k not in Simulator.contention_link:
                Simulator.contention_link[k] = v
            else:
                Simulator.contention_link[k] += v

    if joint_scheduler in ['oxc_scheduler', 'static_scheduler', 'GPUPlacementer', 'GPUPlacementer2',
                           'OCSExpander','OCSExpander_superPod', 'ELEExpander','ELEExpander_SuperPod', 'NaiveScheduler', 'NaiveSchedulerCompare',
                           'GPUPlacementer3', 'GPUPlacementer4', 'StaticPlacementer',
                           'StaticPlacementerRelax', 'hw_oxc_all2all', 'hw_oxc_all2all_sz',
                           'hw_oxc_all2all2', 'hw_oxc_allreduce', 'hw_oxc_hdallreduce',
                           'StaticPlacementerAI']:
        # Update topology and path dict
        if not_need_refresh:
            # 这里虽然没有传入-2，但是在reconfigure函数里面进行了更改。
            Simulator.reconfigure(allocated_link_mapping, task_id)
        else:
            # 这里列出的joint_scheduler是vclos中分配单独子网的情况
            if joint_scheduler not in ['GPUPlacementer', 'GPUPlacementer2', 'GPUPlacementer3', 'GPUPlacementer4',
                                       'StaticPlacementer', 'StaticPlacementerRelax', 'StaticPlacementerAI']:
                raise ValueError("Scheduler type error: {}".format(joint_scheduler))
            infra = Simulator.get_infrastructure()
            infra.refresh_link_flow_occupy_dict()
            for temp_task_id in allocated_link_mapping:
                Simulator.reconfigure(allocated_link_mapping[temp_task_id], temp_task_id)
                if temp_task_id != task_id:
                    infra.update_flow_route_info(temp_task_id)
        # print("finish reconfig")
    if joint_scheduler in ['static', 'static_scheduler', 'static_scheduler_small',
                           'GPUPlacementer3', 'GPUPlacementer2', 'NaiveScheduler',
                           'NaiveSchedulerCompare', 'GPUPlacementer4', 'StaticPlacementer',
                           'StaticPlacementerRelax', 'StaticPlacementerAI']:
        task_type_obj.deal_job(task_id=task_id, model_size=model_size, task_occupied_NIC_num=task_occupied_NIC_num,
                               use_NIC_list=gpu_indexes, NIC_num_in_a_server=NIC_num_in_a_server,
                               special_pair=special_pair)
    elif joint_scheduler in ['OCSExpander','OCSExpander_superPod', 'ELEExpander' ,'ELEExpander_SuperPod']:
        task_type_obj.deal_job(task_id=task_id, model_size=model_size, task_occupied_NIC_num=task_occupied_NIC_num,
                               ep_qp=ep_qp, dp_qp=dp_qp, pp_qp=pp_qp)
    else:
        task_type_obj.deal_job(task_id=task_id, model_size=model_size, task_occupied_NIC_num=task_occupied_NIC_num,
                               use_NIC_list=gpu_indexes, NIC_num_in_a_server=NIC_num_in_a_server)
    return True, gpu_indexes
