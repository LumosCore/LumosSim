
import math
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

class Butterfly(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

        communication_size = model_size

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

        roundid = 0
        for pair_list in round_pair_list:
            # Every round
            for (src, dst) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                            communication_size, None, taskid, roundid, task_occupied_NIC_num)
                self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                Simulator.FLOWID += 1
            roundid += 1
            # TODO: Add computation time

        # Register first round job flows
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(0, flow_list))

    
    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        return self.get_butterfly_every_round_pair(task_occupied_NIC_num)


    def get_butterfly_every_round_pair(self, task_occupied_NIC_num):
        """Return communication pair in every round under butterfly strategy.
        [
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)] ...
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)], ...
            ...
        ]
        """
        butterfly_pair_list = []
        round_num = math.log2(task_occupied_NIC_num)
        assert(round_num.is_integer())
        round_num = int(round_num)
    
        mask = 1
        for _ in range(0, round_num):
            a_round = []
            for pair in range(0, task_occupied_NIC_num):
                NIC_src = pair
                NIC_dst = (pair ^ mask)
                a_round.append((NIC_src, NIC_dst))
            butterfly_pair_list.append(a_round)
            mask = mask * 2

        return butterfly_pair_list


    # def butterfly_primitive(self, node_num):
    #     node_comm_task = []
    #     for src in range(node_num):
    #         step = 2
    #         dst = -1
    #         while(step <= node_num):
    #             if(src % step < step / 2):
    #                 dst = src + step / 2
    #             else:
    #                 dst = src - step / 2
    #             node_comm_task.append([src, int(dst)])
    #             step = step * 2
    #     return node_comm_task