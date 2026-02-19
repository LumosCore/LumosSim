
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

class HwOxcAll2All2(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

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

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

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
                                communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                    self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                    Simulator.FLOWID += 1
                if roundid == max_roundid:
                    flag = True
                roundid += 1


        # Register first round job flows
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(0, flow_list))


    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        if task_occupied_NIC_num <= NIC_num_in_a_server:
            round_pair_list = self.get_all_to_all_every_round_pair(task_occupied_NIC_num, model_size)
        else:
            import numpy
            topology = numpy.zeros([task_occupied_NIC_num, task_occupied_NIC_num])
            if task_occupied_NIC_num == NIC_num_in_a_server * NIC_num_in_a_server:
                for src_node in range(0, NIC_num_in_a_server):
                    for src_nic in range(0, NIC_num_in_a_server):
                        if src_node == src_nic:
                            continue
                        src = src_node * NIC_num_in_a_server + src_nic
                        dst = src_nic * NIC_num_in_a_server + src_node
                        topology[src][dst] = 1
            else:
                print("Need a topology")
                exit(0)
            round_pair_list = self.get_oxc_all2all_every_round_pair(task_occupied_NIC_num, model_size, NIC_num_in_a_server, topology)
        return round_pair_list


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


    # topology is a task_occupied_NIC_num X task_occupied_NIC_num matrix
    # topology[src][dst] is capacity
    def get_oxc_all2all_every_round_pair(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server, topology):
        pair_list = []
        node_num = int(task_occupied_NIC_num / NIC_num_in_a_server)
        src_intermediate_nic_to_dst = self._bind_flow_to_nic_based_on_topology(node_num, NIC_num_in_a_server, topology)

        # intra-node all_to_all
        communication_size = model_size / task_occupied_NIC_num
        round_num = NIC_num_in_a_server - 1
        mask = 1
        for _ in range(0, round_num):
            a_round = []
            for pair in range(0, NIC_num_in_a_server):
                for node_id in range(node_num):
                    NIC_src = pair + node_id * NIC_num_in_a_server
                    NIC_dst = (pair ^ mask) + node_id * NIC_num_in_a_server
                    num_data_segments = 1
                    if pair ^ mask in src_intermediate_nic_to_dst[NIC_src]:
                        num_data_segments += len(src_intermediate_nic_to_dst[NIC_src][pair ^ mask])
                    a_round.append((NIC_src, NIC_dst, communication_size * num_data_segments))
            pair_list.append(a_round)
            mask = mask + 1

        # inter-node all_to_all
        round_list = []
        for src in src_intermediate_nic_to_dst:
            node_id = src // NIC_num_in_a_server
            for nic in src_intermediate_nic_to_dst[src]:                
                intermediate_nic = node_id * NIC_num_in_a_server + nic
                for dst in src_intermediate_nic_to_dst[src][nic]:
                    round_list.append((intermediate_nic, dst, communication_size))
        pair_list.append(round_list)

        return pair_list


    def _bind_flow_to_nic_based_on_topology(self, node_num, NIC_num_in_a_server, topology):
        # 拓扑感知
        src_intermediate_nic_to_dst = {}
        for src_node in range(0, node_num):
            # 初始化
            for src_intra_node_id in range(0, NIC_num_in_a_server):
                src = src_intra_node_id + src_node * NIC_num_in_a_server
                src_intermediate_nic_to_dst[src] = {}
            # 计算
            for dst_node in range(0, node_num):
                if src_node == dst_node:
                    continue
                available_links = {}
                allocation = {}
                # 找出所有与dst_node有连线的网卡
                for src_intra_node_id in range(0, NIC_num_in_a_server):
                    src = src_intra_node_id + src_node * NIC_num_in_a_server
                    for dst_intra_node_id in range(0, NIC_num_in_a_server):
                        dst = dst_intra_node_id + dst_node * NIC_num_in_a_server
                        if topology[src][dst] > 0:
                            assert (src_intra_node_id not in available_links)
                            available_links[src_intra_node_id] = topology[src][dst]
                            allocation[src_intra_node_id] = 0
                            for src2 in range(src_node * NIC_num_in_a_server, (src_node + 1) * NIC_num_in_a_server):
                                src_intermediate_nic_to_dst[src2][src_intra_node_id] = []
                # 为每个通信对分配intermediate nic
                for src_intra_node_id in range(0, NIC_num_in_a_server):
                    src = src_intra_node_id + src_node * NIC_num_in_a_server
                    for dst_intra_node_id in range(0, NIC_num_in_a_server):
                        dst = dst_intra_node_id + dst_node * NIC_num_in_a_server
                        if src_intra_node_id in available_links:
                            src_intermediate_nic_to_dst[src][src_intra_node_id].append(dst)
                            allocation[src_intra_node_id] += 1
                        else:
                            least_occupied_nic = -1
                            min_utilization = 1e6
                            for nic in available_links:
                                if allocation[nic] / available_links[nic] < min_utilization:
                                    min_utilization = allocation[nic] / available_links[nic]
                                    least_occupied_nic = nic
                            src_intermediate_nic_to_dst[src][least_occupied_nic].append(dst)
                            allocation[least_occupied_nic] += 1
        return src_intermediate_nic_to_dst

if __name__ == '__main__':
    test = HwOxcAll2All2()
    res = test.get_oxc_all2all_every_round_pair(16, 220, 4)
    print(res)
