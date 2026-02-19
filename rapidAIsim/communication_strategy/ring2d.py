import math
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow


class Ring(StrategyBase):
    def __init__(self) -> None:
        super().__init__()

    def is_power_of_2(self, n):
        return n & (n - 1) == 0

    def closest_power_of_2(self, initial):
        temp = initial - 1
        temp |= temp >> 1
        temp |= temp >> 2
        temp |= temp >> 4
        temp |= temp >> 8
        temp |= temp >> 16
        target = 1 if temp < 0 else temp + 1
        return target

    def closest_min_power_of_2(self, initial):
        closest_power_num = self.closest_power_of_2(initial)
        if closest_power_num > initial:
            return int(2 ** (math.log(closest_power_num, 2) - 1))
        else:
            return int(closest_power_num)

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair=None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

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

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server,
                                                              use_NIC_list, special_pair)

        roundid = 0
        for pair_list in round_pair_list:
            # Every round
            for (src, dst, communication_size) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                            communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                Simulator.FLOWID += 1
            roundid += 1

        # Register first round job flows
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(computation_time, flow_list))

    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server, use_NIC_list=[],
                                       special_pair=None):
        if task_occupied_NIC_num <= NIC_num_in_a_server:
            round_pair_list = self.get_ring_every_round_pair(task_occupied_NIC_num, model_size)
        elif not self.is_power_of_2(task_occupied_NIC_num):
            # closest_min_power_num = self.closest_min_power_of_2(task_occupied_NIC_num)
            # tmp = self.get_ring_every_round_pair(closest_min_power_num, model_size)
            # before_extra_pair_list = []
            # after_extra_pair_list = []
            # diff_num = task_occupied_NIC_num - closest_min_power_num
            # if special_pair == None:
            #     return [[(0, 0, 0)]] + tmp + [[(0,0, 0)]]
            # for src, dst in special_pair:
            #     before_extra_pair_list.append((src, dst, model_size))
            #     after_extra_pair_list.append((dst, src, model_size))
            # return [before_extra_pair_list] + tmp + [after_extra_pair_list]
            round_pair_list = self.get_multi_server_every_round_pair(task_occupied_NIC_num, model_size,
                                                                     NIC_num_in_a_server)
        else:
            round_pair_list = self.get_multi_server_every_round_pair(task_occupied_NIC_num, model_size,
                                                                     NIC_num_in_a_server)
        return round_pair_list

    def get_ring_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under ring strategy.
        [
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)] ...
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)], ...
            ...
        ]
        """
        ring_pair_list = []
        round_num = 2 * (task_occupied_NIC_num - 1)
        communication_size = model_size / task_occupied_NIC_num / 2
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

    def get_multi_server_every_round_pair(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        pair_list = []
        node_num = int(task_occupied_NIC_num / NIC_num_in_a_server)

        # ring allreduce in the intra-server
        round_num = NIC_num_in_a_server - 1
        communication_size = model_size / NIC_num_in_a_server
        for _ in range(round_num):
            forward = []
            for i in range(NIC_num_in_a_server):
                src = i
                if i == NIC_num_in_a_server - 1:
                    dst = 0
                else:
                    dst = i + 1
                for j in range(node_num):
                    forward.append((src + j * NIC_num_in_a_server, dst + j * NIC_num_in_a_server, communication_size))
            # print(forward)
            pair_list.append(forward)

        # ring allreduce inter servers
        communication_size = model_size / NIC_num_in_a_server / node_num
        round_num = node_num
        for _ in range(round_num):
            forward = []
            for i in range(round_num):
                src = i
                if i == node_num - 1:
                    dst = 0
                else:
                    dst = i + 1
                for j in range(NIC_num_in_a_server):
                    forward.append((src * NIC_num_in_a_server + j, dst * NIC_num_in_a_server + j, communication_size))
            # print(forward)
            pair_list.append(forward)

        # ring allreduce in the intra-server
        communication_size = model_size / NIC_num_in_a_server
        round_num = NIC_num_in_a_server - 1
        for _ in range(round_num):
            forward = []
            for i in range(NIC_num_in_a_server):
                src = i
                if i == NIC_num_in_a_server - 1:
                    dst = 0
                else:
                    dst = i + 1
                for j in range(node_num):
                    forward.append((src + j * NIC_num_in_a_server, dst + j * NIC_num_in_a_server, communication_size))
            # print(forward)
            pair_list.append(forward)

        return pair_list

    def get_expected_completion_time(self, task_seq=0):
        pass
