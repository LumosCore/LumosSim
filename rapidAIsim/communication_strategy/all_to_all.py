
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

class AllToAll(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair = None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

        conservative = False
        if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
            conservative = True
        computation_time = Simulator.TASK_LIST[taskid].computation_time

        # Deal with only 1 GPU occupation
        if task_occupied_NIC_num == 1:
            flow_list = []
            flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None, taskid,
                        0, task_occupied_NIC_num, conservative)
            self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
            flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))
            Simulator.FLOWID += 1
            return

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

        roundid = 0
        flag = False

        for pair_list in round_pair_list:
            # Every round
            for (src, dst, communication_size) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                            communication_size, None, taskid, roundid, task_occupied_NIC_num, conservative)
                self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                Simulator.FLOWID += 1

            roundid += 1
        # roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
        # max_roundid = int(roundidflag_list[-1])
        # flag = False
        # while flag == False:
        #     for pair_list in round_pair_list:
        #         # Every round
        #         for (src, dst, communication_size) in pair_list:
        #             # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
        #             flow = Flow(
        #                 Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
        #                 communication_size, None,
        #                 taskid, roundid, task_occupied_NIC_num, conservative
        #             )
        #             self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
        #             Simulator.FLOWID += 1
        #         if roundid == max_roundid:
        #             flag = True
        #         roundid += 1


        # Register first round job flows
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))

    
    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        return self.get_all_to_all_every_round_pair(task_occupied_NIC_num, model_size)


    def get_all_to_all_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under all-to-all strategy.
        [
            [(NIC_src, NIC_dst, communication_size), (NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size), (NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        all_to_all_pair_list = []
        
        round_num = task_occupied_NIC_num - 1
    
        # all_to_all
        mask = 1
        communication_size = model_size / task_occupied_NIC_num
        for _ in range(0, round_num):
            a_round = []
            for pair in range(0, task_occupied_NIC_num):
                NIC_src = pair
                NIC_dst = (pair ^ mask)
                a_round.append((NIC_src, NIC_dst, communication_size))
            all_to_all_pair_list.append(a_round)
            mask = mask + 1
        return all_to_all_pair_list


if __name__ == '__main__':
    test = AllToAll()
    res = test.get_all_to_all_every_round_pair(8, 80)
    for round in res:
        print(round)
