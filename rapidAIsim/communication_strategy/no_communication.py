import math
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow


class NoCommunication(StrategyBase):
    def __init__(self) -> None:
        super().__init__()

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair = None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')
        computation_time = Simulator.TASK_LIST[taskid].computation_time
        conservative = False
        if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
            conservative = True

        flow_list = []
        flow = Flow(Simulator.FLOWID, model_size, None, 0, 0, model_size, None, taskid, 0, task_occupied_NIC_num,
                    conservative)
        self.record_network_occupy(taskid, 0, flow, 0)
        flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))
        Simulator.FLOWID += 1
        return

    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server, special_pair = None):
        return self.get_butterfly2_every_round_pair(task_occupied_NIC_num, model_size)

    def get_butterfly2_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under butterfly strategy.
        [
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        butterfly_pair_list = []
        
        final_butterfly_pair_list = butterfly_pair_list.copy()
        length = len(butterfly_pair_list)
        for i in range(length - 1, -1, -1):
            final_butterfly_pair_list.append(butterfly_pair_list[i])

        return final_butterfly_pair_list

    def is_power_of_2(self, n):
        return n & (n - 1) == 0







