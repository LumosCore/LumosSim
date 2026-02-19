from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow
import math
import random


class All2All_OXC(StrategyBase):
    def __init__(self, random_factor_value=0):
        super().__init__()
        self.random_factor_value = random_factor_value

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair = None, TP_size=8):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent
        use_NIC_list.sort()
        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        print("debug_used_nic")
        print(use_NIC_list)
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')
        computation_time = Simulator.TASK_LIST[taskid].computation_time

        # Deal with only 1 GPU occupation
        if task_occupied_NIC_num == 1:
            flow_list = []
            flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None, taskid,
                        0, task_occupied_NIC_num, False)
            self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
            flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))
            Simulator.FLOWID += 1
            return

        communication_size = model_size

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, communication_size, NIC_num_in_a_server, use_NIC_list, TP_size)

        roundid = 0
        comm_pair_set = {}
        for pair_list in round_pair_list:
            # Every round
            for (src, dst, communication_size2) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                tmp_communication_size = communication_size2
                flow = Flow(Simulator.FLOWID, tmp_communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                            tmp_communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                key = int(use_NIC_list[src]/NIC_num_in_a_server)
                value = (use_NIC_list[src], use_NIC_list[dst], roundid)
                if key not in comm_pair_set:
                    comm_pair_set[key] = []
                comm_pair_set[key].append(value)
                self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                Simulator.FLOWID += 1
            roundid += 1

        # Register first round job flow
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))
        
    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, communication_size, NIC_num_in_a_server, special_pair = None, TP_size=8):
        round_pair_list = self.get_pairwise_every_round_pair(task_occupied_NIC_num, communication_size, TP_size)
        
        return round_pair_list

    @staticmethod
    def get_pairwise_every_round_pair(task_occupied_NIC_num, communication_size, TP_size):
        """Return communication pair in every round under ring strategy.
        [
            [(src_rank, dst_rank, size), (src_rank, dst_rank, size), ...],
            [(src_rank, dst_rank, size), (src_rank, dst_rank, size), ...],
            ...
        ]
        """
        ring_pair_list = []
        EP_size = task_occupied_NIC_num // TP_size
        round_num = (EP_size - 1)

        for i in range(round_num):
            forward = []
            for src in range(task_occupied_NIC_num):
                j = src % TP_size
                dis = (i+j)%(EP_size-1) + 1
                dst = (src + dis * TP_size) % task_occupied_NIC_num
                forward.append((src, dst, communication_size))
            ring_pair_list.append(forward)

        return ring_pair_list
    
    @staticmethod
    def get_pairwise_every_round_pair_old(task_occupied_NIC_num, communication_size):
        """Return communication pair in every round under ring strategy.
        [
            [(src_rank, dst_rank, size), (src_rank, dst_rank, size), ...],
            [(src_rank, dst_rank, size), (src_rank, dst_rank, size), ...],
            ...
        ]
        """
        ring_pair_list = []
        round_num = (task_occupied_NIC_num - 1)

        for i in range(round_num):
            forward = []
            for src in range(task_occupied_NIC_num):
                dst = (src + i + 1) % task_occupied_NIC_num
                forward.append((src, dst, communication_size))
            ring_pair_list.append(forward)

        return ring_pair_list
    

if __name__ == '__main__':
    test = All2All_OXC()
    res = test.get_pairwise_every_round_pair(12, 10, 3)
    for round in res:
        print(round)
    print()
    res = test.get_pairwise_every_round_pair_old(12, 10)
    for round in res:
        print(round)