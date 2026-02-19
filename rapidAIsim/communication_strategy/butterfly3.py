import math
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow


class Butterfly3(StrategyBase):
    def __init__(self) -> None:
        pass

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
        from rapidAIsim.core.simulator import Simulator

        computation_time = Simulator.TASK_LIST[taskid].computation_time
        schedule_time_cost = Simulator.SCHEDULER_TIME_COST[taskid]
        if task_occupied_NIC_num == pow(2, math.ceil(math.log2(task_occupied_NIC_num))):
            """The initial jobs are assigned according to communication strategy.
            """
            from rapidAIsim.core.simulator import Simulator
            from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

            print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
            Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

            conservative = Simulator.CONF_DICT['find_next_hop_method'] == 'conservative'

            # Deal with only 1 GPU occupation
            if task_occupied_NIC_num == 1:
                flow_list = []
                flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None,
                            taskid, 0, task_occupied_NIC_num, conservative)
                self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
                flow_list.append(flow)
                Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))
                Simulator.FLOWID += 1
                return

            print("debug NIC_num_in_a_server", NIC_num_in_a_server)
            round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size,
                                                                  NIC_num_in_a_server)

            roundid = 0
            roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
            max_roundid = int(roundidflag_list[-1])

            flag = False
            while flag == False:
                for pair_list in round_pair_list:
                    # Every round
                    for (src, dst, communication_size) in pair_list:
                        # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                        flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                                    communication_size, None, taskid, roundid, task_occupied_NIC_num, conservative)
                        self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                        Simulator.FLOWID += 1
                    if roundid == max_roundid:
                        flag = True
                    roundid += 1

            # Register first round job flows
            flow_list = []
            for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
                flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))
        else:
            """The initial jobs are assigned according to communication strategy.
            """
            from rapidAIsim.core.simulator import Simulator
            from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

            print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
            Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

            # Deal with only 1 GPU occupation
            if task_occupied_NIC_num == 1:
                flow_list = []
                flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None,
                            taskid, 0, task_occupied_NIC_num, False)
                self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
                flow_list.append(flow)
                Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))
                Simulator.FLOWID += 1
                return

            communication_size = model_size / task_occupied_NIC_num / 2

            round_pair_list = self.get_butterfly3_every_round_pair(task_occupied_NIC_num, model_size)

            roundid = 0
            roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
            max_roundid = int(roundidflag_list[-1])  # For supporting multiple interation

            flag = False
            while flag == False:
                for pair_list in round_pair_list:
                    # Every round
                    for (src, dst) in pair_list:
                        # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                        flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                                    communication_size, None, taskid, roundid, task_occupied_NIC_num)
                        self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                        Simulator.FLOWID += 1
                    if roundid == max_roundid:
                        flag = True
                    roundid += 1

            # Register first round job flow
            flow_list = []
            for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
                flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))

    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        round_pair_list = self.get_butterfly3_every_round_pair(task_occupied_NIC_num, model_size)
        return round_pair_list

    def get_butterfly3_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under butterfly strategy.
        [
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        if (task_occupied_NIC_num > 8):
            butterfly_pair_list = []
            round_num = math.log2(task_occupied_NIC_num)
            assert (round_num.is_integer())
            round_num = int(round_num)

            # Reduce-Scatter
            mask = 1
            communication_size = model_size / 2
            for _ in range(0, round_num):
                a_round = []
                for pair in range(0, task_occupied_NIC_num):
                    NIC_src = pair
                    NIC_dst = (pair ^ mask)
                    a_round.append((NIC_src, NIC_dst, communication_size))
                butterfly_pair_list.append(a_round)
                mask = mask * 2
                communication_size = communication_size / 2

            # All-Gather
            # ---- error ----
            # mask = 1
            # communication_size = model_size / task_occupied_NIC_num
            # for _ in range(0, round_num):
            #     a_round = []
            #     for pair in range(0, task_occupied_NIC_num):
            #         NIC_src = pair
            #         NIC_dst = (pair ^ mask)
            #         a_round.append((NIC_src, NIC_dst, communication_size))
            #     butterfly_pair_list.append(a_round)
            #     mask = mask * 2
            #     communication_size = communication_size * 2
            # ---- error ----
            final_butterfly_pair_list = butterfly_pair_list.copy()
            length = len(butterfly_pair_list)
            for i in range(length - 1, -1, -1):
                final_butterfly_pair_list.append(butterfly_pair_list[i])
            return final_butterfly_pair_list
        else:
            ring_pair_list = []
            round_num = 2 * (task_occupied_NIC_num - 1)

            for _ in range(round_num):
                forward = []
                backward = []
                for i in range(task_occupied_NIC_num):
                    src = i
                    if i == task_occupied_NIC_num - 1:
                        dst = 0
                    else:
                        dst = i + 1
                    forward.append((src, dst))
                    backward.append((dst, src))
                ring_pair_list.append(forward + backward)

            return ring_pair_list

    # def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
    #     """The initial jobs are assigned according to communication strategy.
    #     """
    #     from rapidAIsim.core.simulator import Simulator
    #     from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

    #     print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
    #     Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')
    #     #computation_time = float(Simulator.CONF_DICT['computation_time'])
    #     computation_time = float(Simulator.TASK_LIST[taskid][3])
    #     conservative = False
    #     if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
    #         conservative = True

    #     schedule_time_cost = Simulator.SCHEDULER_TIME_COST[taskid]
    #     # Deal with only 1 GPU occupation
    #     if task_occupied_NIC_num == 1:
    #         flow_list = []
    #         flow = Flow(
    #             Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0],
    #             model_size, None,
    #             taskid, 0, task_occupied_NIC_num, conservative
    #         )
    #         self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
    #         flow_list.append(flow)
    #         Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))
    #         Simulator.FLOWID += 1
    #         return

    #     round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

    #     roundid = 0
    #     roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
    #     max_roundid = int(roundidflag_list[-1])

    #     flag = False
    #     while flag == False:
    #         for pair_list in round_pair_list:
    #             # Every round
    #             for (src, dst, communication_size) in pair_list:
    #                 # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
    #                 flow = Flow(
    #                     Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
    #                     communication_size, None,
    #                     taskid, roundid, task_occupied_NIC_num, conservative
    #                 )
    #                 self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
    #                 Simulator.FLOWID += 1
    #             if roundid == max_roundid:
    #                 flag = True
    #             roundid += 1

    #     # Register first round job flows
    #     flow_list = []
    #     for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
    #         flow_list.append(flow)
    #     Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))

    # def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
    #     return self.get_butterfly2_every_round_pair(task_occupied_NIC_num, model_size)

    # def get_butterfly2_every_round_pair(self, task_occupied_NIC_num, model_size):
    #     """Return communication pair in every round under butterfly strategy.
    #     [
    #         [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
    #         [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
    #         ...
    #     ]
    #     """
    #     butterfly_pair_list = []
    #     round_num = math.log2(task_occupied_NIC_num)
    #     assert(round_num.is_integer())
    #     round_num = int(round_num)

    #     # Reduce-Scatter
    #     mask = 1
    #     communication_size = model_size / 2
    #     for _ in range(0, round_num):
    #         a_round = []
    #         for pair in range(0, task_occupied_NIC_num):
    #             NIC_src = pair
    #             NIC_dst = (pair ^ mask)
    #             a_round.append((NIC_src, NIC_dst, communication_size))
    #         butterfly_pair_list.append(a_round)
    #         mask = mask * 2
    #         communication_size = communication_size / 2

    #     # All-Gather
    #     # ---- error ----
    #     # mask = 1
    #     # communication_size = model_size / task_occupied_NIC_num
    #     # for _ in range(0, round_num):
    #     #     a_round = []
    #     #     for pair in range(0, task_occupied_NIC_num):
    #     #         NIC_src = pair
    #     #         NIC_dst = (pair ^ mask)
    #     #         a_round.append((NIC_src, NIC_dst, communication_size))
    #     #     butterfly_pair_list.append(a_round)
    #     #     mask = mask * 2
    #     #     communication_size = communication_size * 2
    #     # ---- error ----
    #     final_butterfly_pair_list = butterfly_pair_list.copy()
    #     length = len(butterfly_pair_list)
    #     for i in range(length - 1, -1, -1):
    #         final_butterfly_pair_list.append(butterfly_pair_list[i])

    #     return final_butterfly_pair_list
