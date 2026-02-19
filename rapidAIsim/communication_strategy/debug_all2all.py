
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

class All2All(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair = None, GPU_each_leaf = 16):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent
        use_NIC_list.sort()
        # use_NIC_list = []
        # for tmp_leaf_id in range(16):
        #     use_NIC_list.append(tmp_leaf_id*GPU_each_leaf)
        #     use_NIC_list.append(tmp_leaf_id*GPU_each_leaf+1)
        #     use_NIC_list.append(tmp_leaf_id*GPU_each_leaf+2)
        #     use_NIC_list.append(tmp_leaf_id*GPU_each_leaf+3)
        # use_NIC_list = [i for i in range(task_occupied_NIC_num)]
        print(f'Time {0} start task {taskid} occuping NIC num {len(use_NIC_list)}')

        leaf_gpu_map = {}
        for tmp_gpu_id in use_NIC_list:
            src_leaf = int(tmp_gpu_id/GPU_each_leaf)
            if src_leaf not in leaf_gpu_map:
                leaf_gpu_map[src_leaf] = 0
            leaf_gpu_map[src_leaf] += 1
        print(leaf_gpu_map)
        communication_size = model_size
        # TODO
        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, communication_size, NIC_num_in_a_server, use_NIC_list)

        roundid = 0
        for pair_list in round_pair_list:
            # Every round
            print(roundid)
            spine_be_aimed_map = {}
            for (src, dst, communication_size2) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                assert communication_size2 == communication_size
                # if taskid == 1247 and roundid == 16:
                #     print("debug flow",use_NIC_list[src], use_NIC_list[dst],roundid)
                src_leaf = int(use_NIC_list[src]/GPU_each_leaf)
                inter_spine = use_NIC_list[src]%GPU_each_leaf
                dst_leaf = int(use_NIC_list[dst]/GPU_each_leaf)
                if inter_spine not in spine_be_aimed_map:
                    spine_be_aimed_map[inter_spine] = {}
                if dst_leaf not in spine_be_aimed_map[inter_spine]:
                    spine_be_aimed_map[inter_spine][dst_leaf] = []
                if src_leaf not in spine_be_aimed_map[inter_spine][dst_leaf]:
                    spine_be_aimed_map[inter_spine][dst_leaf].append(src_leaf)
            for inter_spine in spine_be_aimed_map:
                for dst_leaf in spine_be_aimed_map[inter_spine]:
                    assert len( spine_be_aimed_map[inter_spine][dst_leaf])<=1
            print(spine_be_aimed_map)
            roundid += 1



    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, communication_size, NIC_num_in_a_server, special_pair = None):
        round_pair_list = self.get_pairwise_every_round_pair(task_occupied_NIC_num, communication_size)
        return round_pair_list


    def get_pairwise_every_round_pair(self, task_occupied_NIC_num, communication_size):
        """Return communication pair in every round under ring strategy.
        [
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)] ...
            [(NIC_src, NIC_dst)], [(NIC_src, NIC_dst)], ...
            ...
        ]
        """
        ring_pair_list = []
        round_num = (task_occupied_NIC_num - 1)

        for _ in range(round_num):
            forward = []
            for i in range(task_occupied_NIC_num):
                src = i
                dst = (i+_+1)%task_occupied_NIC_num
                forward.append((src, dst, communication_size))
            ring_pair_list.append(forward )

        return ring_pair_list


    
if __name__ == '__main__':
    test = All2All()
    gpu_list = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 316, 317, 318, 319, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 468, 469, 470, 471, 500, 501, 502, 503, 504, 505, 506, 507]
    test.deal_job(1247, 1000, 64, gpu_list, 4)
    # res = test.get_pairwise_every_round_pair(4, 10)
    # for round in res:
    #     print(round)