
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

class Ring(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair = None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent
        print(use_NIC_list)
        use_NIC_list.sort()
        print(use_NIC_list)
        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

        # Deal with only 1 GPU occupation
        if task_occupied_NIC_num == 1:
            flow_list = []
            flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None, taskid,
                        0, task_occupied_NIC_num, False)
            self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
            flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(0, flow_list))
            Simulator.FLOWID += 1
            return

        communication_size = model_size / task_occupied_NIC_num / 2

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, communication_size, NIC_num_in_a_server, use_NIC_list)

        roundid = 0
        for pair_list in round_pair_list:
            # Every round
            for (src, dst, communication_size2) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                assert communication_size2 == communication_size
                flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                            communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                Simulator.FLOWID += 1
            roundid += 1

        # Register first round job flow
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(0, flow_list))


    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, communication_size, NIC_num_in_a_server, use_NIC_list):
        round_pair_list = self.get_ring_every_round_pair(task_occupied_NIC_num, communication_size)
        return round_pair_list


    def get_ring_every_round_pair(self, task_occupied_NIC_num, communication_size):
        """Return communication pair in every round under ring strategy.
        [
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)] ...
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)], ...
            ...
        ]
        """
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
                forward.append((src, dst, communication_size))
                backward.append((dst, src, communication_size))
            ring_pair_list.append(forward + backward)

        return ring_pair_list

    def get_expected_completion_time(self, task_seq = 0):
        from rapidAIsim.core.simulator import Simulator
        task = Simulator.TASK_LIST[task_seq]
        model_size, task_occupied_NIC_num = task.model_size, task.gpu_num
        NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        sync_delay_alpha = float(Simulator.CONF_DICT['sync_delay_alpha'])
        include_server_sync_delay_alpha = float(Simulator.CONF_DICT['include_server_sync_delay_alpha'])
        if task_occupied_NIC_num == 1:
            intra_datafactor = 1
            inter_datafactor = 0
        elif task_occupied_NIC_num <= NIC_num_in_a_server:
            intra_datafactor = 2 * (1 - 1 / task_occupied_NIC_num)
            inter_datafactor = 0
        else:
            intra_datafactor = 0
            inter_datafactor = 1 - 1 / task_occupied_NIC_num
        if task_occupied_NIC_num == 1:
            intra_times = 0
            inter_times = 0
        elif task_occupied_NIC_num <= NIC_num_in_a_server:
            intra_times = 4 * (task_occupied_NIC_num - 1)
            inter_times = 0
        else:
            intra_times = 0
            inter_times = 2 * (task_occupied_NIC_num - 1)
        expected_completion_time = model_size * (intra_datafactor / float(Simulator.CONF_DICT['inner_server_bandwidth']) + inter_datafactor / float(Simulator.CONF_DICT['switch_port_bandwidth'])) + intra_times * include_server_sync_delay_alpha + inter_times * sync_delay_alpha
        return expected_completion_time
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow