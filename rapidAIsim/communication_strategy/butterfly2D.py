import math
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow


class Butterfly2D(StrategyBase):
    def __init__(self) -> None:
        super().__init__()

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair=None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

        # print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')
        computation_time = Simulator.TASK_LIST[taskid].computation_time
        conservative = False
        if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
            conservative = True

        schedule_time_cost = Simulator.SCHEDULER_TIME_COST[taskid]
        # Deal with only 1 GPU occupation
        if task_occupied_NIC_num == 1:
            flow_list = []
            flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None, taskid,
                        0, task_occupied_NIC_num, conservative)
            self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
            flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))
            Simulator.FLOWID += 1
            return

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server,
                                                              special_pair)

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
        Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))

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

    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server, special_pair=None):
        if self.is_power_of_2(task_occupied_NIC_num):
            return self.get_butterfly2D_every_round_pair(task_occupied_NIC_num, model_size, NIC_num_in_a_server)
        else:
            closest_min_power_num = self.closest_min_power_of_2(task_occupied_NIC_num)
            tmp = self.get_butterfly2D_every_round_pair(closest_min_power_num, model_size, NIC_num_in_a_server)
            before_extra_pair_list = []
            after_extra_pair_list = []
            diff_num = task_occupied_NIC_num - closest_min_power_num
            # for i in range(0, diff_num):
            #     before_extra_pair_list.append((task_occupied_NIC_num + i, i, model_size))
            #     after_extra_pair_list.append((i, task_occupied_NIC_num + i, model_size))

            if special_pair is None:
                return [[(0, 0, 0)]] + tmp + [[(0, 0, 0)]]
            for src, dst in special_pair:
                before_extra_pair_list.append((src, dst, model_size))
                after_extra_pair_list.append((dst, src, model_size))
            return [before_extra_pair_list] + tmp + [after_extra_pair_list]

    def get_butterfly2D_every_round_pair(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        """Return communication pair in every round under butterfly strategy.
        [
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        from rapidAIsim.core.simulator import Simulator
        butterfly_pair_list = []

        node_num = max(1, int(task_occupied_NIC_num / NIC_num_in_a_server))

        communication_size = model_size / node_num
        # ring allreduce in the intra-server
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
            butterfly_pair_list.append(forward)

        group_size = max(1, int(task_occupied_NIC_num / NIC_num_in_a_server))
        round_num = math.log2(group_size)
        assert (round_num.is_integer())

        round_num = int(round_num)

        # # Reduce-Scatter
        # mask = 1
        # communication_size = model_size / 2
        # for _ in range(0, round_num):
        #     a_round = []
        #     for pair in range(0, task_occupied_NIC_num):
        #         NIC_src = pair
        #         NIC_dst = (pair ^ mask)
        #         a_round.append((NIC_src, NIC_dst, communication_size))
        #     print(round_num,a_round)
        #     butterfly_pair_list.append(a_round)
        #     mask = mask * 2
        #     communication_size = communication_size / 2

        # final_butterfly_pair_list = butterfly_pair_list.copy()
        # length = len(butterfly_pair_list)
        # for i in range(length - 1, -1, -1):
        #     final_butterfly_pair_list.append(butterfly_pair_list[i])

        # return final_butterfly_pair_list

        # Reduce-Scatter
        mask = 1
        communication_size = model_size / 2 / NIC_num_in_a_server

        for _ in range(0, round_num):
            a_round = []
            for group_id in range(NIC_num_in_a_server):
                for pair in range(0, group_size):
                    NIC_src = pair * group_size + group_id
                    NIC_dst = (pair ^ mask) * group_size + group_id
                    # print(group_size, group_id, pair, (pair ^ mask),  NIC_src, NIC_dst)
                    a_round.append((NIC_src, NIC_dst, communication_size))
            butterfly_pair_list.append(a_round)
            # print(task_occupied_NIC_num, a_round)
            mask = mask * 2
            communication_size = communication_size / 2

        final_butterfly_pair_list = butterfly_pair_list.copy()
        length = len(butterfly_pair_list)
        for i in range(length - 1, -1, -1):
            final_butterfly_pair_list.append(butterfly_pair_list[i])

        return final_butterfly_pair_list

    def get_expected_completion_time(self, task_seq=0):
        from rapidAIsim.core.simulator import Simulator
        task = Simulator.TASK_LIST[task_seq]
        model_size, task_occupied_NIC_num = task.model_size, task.gpu_num
        NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        node_num = int(task_occupied_NIC_num / NIC_num_in_a_server)
        expected_intra_time = 2 * model_size * (NIC_num_in_a_server - 1) / (
                    NIC_num_in_a_server * float(Simulator.CONF_DICT['inner_server_bandwidth']))
        expected_inter_time = 2 * model_size * (node_num - 1) / (
                    task_occupied_NIC_num * float(Simulator.CONF_DICT['switch_port_bandwidth']))
        expected_completion_time = expected_intra_time + expected_inter_time
        return expected_completion_time
