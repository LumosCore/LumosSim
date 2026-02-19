from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow
import math
import random


class All2All(StrategyBase):
    def __init__(self, random_factor_value=0):
        super().__init__()
        self.random_factor_value = random_factor_value

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair = None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent
        use_NIC_list.sort()
        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
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
        # TODO
        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, communication_size, NIC_num_in_a_server, use_NIC_list)

        roundid = 0
        comm_pair_set = {}
        for pair_list in round_pair_list:
            # Every round
            for (src, dst, communication_size2) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id

                if 'flowletsize' not in Simulator.CONF_DICT or  Simulator.CONF_DICT['flowletsize'] == 'MAX' or Simulator.CONF_DICT['flowletsize'] == '' or int(use_NIC_list[src]/NIC_num_in_a_server) == int(use_NIC_list[dst]/NIC_num_in_a_server):
                    random_factor_value = self.random_factor_value
                    if 'random_factor' in Simulator.CONF_DICT and Simulator.CONF_DICT['random_factor'] != '':
                        random_factor_value = float(Simulator.CONF_DICT['random_factor'])
                    # tmp_communication_size = communication_size2*random.uniform(0.8,1.2)
                    # tmp_communication_size = communication_size2*random.uniform(0.5,1.5)
                    # tmp_communication_size = communication_size2*random.uniform(0.2,1.8)
                    # tmp_communication_size = communication_size2
                    tmp_communication_size = communication_size2*random.uniform(max(0,1-random_factor_value),1+random_factor_value)
                    flow = Flow(Simulator.FLOWID, tmp_communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                                tmp_communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                    key = int(use_NIC_list[src]/NIC_num_in_a_server)
                    value = (use_NIC_list[src], use_NIC_list[dst], roundid)
                    if key not in comm_pair_set:
                        comm_pair_set[key] = []
                    comm_pair_set[key].append(value)
                    self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                    Simulator.FLOWID += 1
                else:
                    flowletsize = float(Simulator.CONF_DICT['flowletsize'])
                    flowlet_num = min(10,math.ceil(communication_size/flowletsize))
                    flowletsize = communication_size/flowlet_num
                    remain_size = communication_size
                    # print("debug remain_size",remain_size,flowletsize,flowlet_num)
                    while remain_size>0:
                        flow = Flow(Simulator.FLOWID, min(remain_size, flowletsize), None, use_NIC_list[src],
                                    use_NIC_list[dst], min(remain_size, flowletsize), None, taskid, roundid,
                                    task_occupied_NIC_num, False)
                        key = int(use_NIC_list[src]/NIC_num_in_a_server)
                        value = (use_NIC_list[src], use_NIC_list[dst], roundid)
                        if key not in comm_pair_set:
                            comm_pair_set[key] = []
                        comm_pair_set[key].append(value)
                        self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                        Simulator.FLOWID += 1
                        remain_size = remain_size - flowletsize
            roundid += 1

        # Register first round job flow
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))
        
        # print("debug comm_set")
        # print(comm_pair_set)

    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, communication_size, NIC_num_in_a_server, special_pair = None):
        round_pair_list = self.get_pairwise_every_round_pair(task_occupied_NIC_num, communication_size)
        # print("debug get_expected_completion_time", self.get_expected_completion_time(communication_size, NIC_num_in_a_server, task_occupied_NIC_num))
        return round_pair_list

    @staticmethod
    def get_pairwise_every_round_pair(task_occupied_NIC_num, communication_size):
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

    def get_expected_completion_time(self, model_size, NIC_num_in_a_server, task_occupied_NIC_num):
        from rapidAIsim.core.simulator import Simulator
        if task_occupied_NIC_num == 1:
            intra_datafactor = 1
            inter_datafactor = 0
        elif task_occupied_NIC_num <= NIC_num_in_a_server:
            intra_datafactor = 1
            inter_datafactor = 0
        else:
            # TODO
            intra_datafactor = NIC_num_in_a_server/task_occupied_NIC_num
            inter_datafactor = 1 - NIC_num_in_a_server/task_occupied_NIC_num
        # print("debug inter time", (task_occupied_NIC_num-NIC_num_in_a_server) ,model_size*(task_occupied_NIC_num-NIC_num_in_a_server) / float(Simulator.CONF_DICT['switch_port_bandwidth']))
        expected_completion_time =  model_size * ((task_occupied_NIC_num-1) / float(Simulator.CONF_DICT['switch_port_bandwidth'])) 
        return expected_completion_time
    

    # def get_expected_completion_time(self, task_seq = 0):
    #     from rapidAIsim.core.simulator import Simulator
    #     task_list = Simulator.TASK_LIST
    #     _, model_size, task_occupied_NIC_num = task_list[task_seq]
    #     NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
    #     sync_delay_alpha = float(Simulator.CONF_DICT['sync_delay_alpha'])
    #     include_server_sync_delay_alpha = float(Simulator.CONF_DICT['include_server_sync_delay_alpha'])
    #     if task_occupied_NIC_num == 1:
    #         intra_datafactor = 1
    #         inter_datafactor = 0
    #     elif task_occupied_NIC_num <= NIC_num_in_a_server:
    #         intra_datafactor = 1
    #         inter_datafactor = 0
    #     else:
    #         # TODO
    #         intra_datafactor = NIC_num_in_a_server/task_occupied_NIC_num
    #         inter_datafactor = 1 - NIC_num_in_a_server/task_occupied_NIC_num
    #     expected_completion_time = model_size * (intra_datafactor / float(Simulator.CONF_DICT['inner_server_bandwidth']) + inter_datafactor / float(Simulator.CONF_DICT['switch_port_bandwidth'])) 
    #     return expected_completion_time


if __name__ == '__main__':
    test = All2All()
    # gpu_list = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 316, 317, 318, 319, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 468, 469, 470, 471, 500, 501, 502, 503, 504, 505, 506, 507]
    # test.deal_job(1247, 1000, 32, gpu_list, 4)
    res = test.get_pairwise_every_round_pair(4, 10)
    for round in res:
        print(round)
