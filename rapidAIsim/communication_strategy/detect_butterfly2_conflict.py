
import math

class DetectButterfly2Conflict:
    def __init__(self) -> None:
        pass


    def whether_conflict(self, model_size, task_occupied_NIC_num, use_NIC_list):
        """Detect a new task whether resulting in confliction.
        """
        from rapidAIsim.core.simulator import Simulator

        round_pair_list = self.get_butterfly2_every_round_pair(task_occupied_NIC_num, model_size)
        
        for pair_list in round_pair_list:
            # Every round
            self_conflict = {}
            for (src, dst, communication_size) in pair_list:

                # Detect confliction with other tasks
                hop_list = Simulator.get_hop_list_from_beforehand_path(use_NIC_list[src], use_NIC_list[dst])
                tmp_src = use_NIC_list[src]
                for next_hop in hop_list:
                    if Simulator.LINK_OCCUPIED_FOR_TASKS.get((tmp_src, next_hop)):
                        if len(Simulator.LINK_OCCUPIED_FOR_TASKS.get((tmp_src, next_hop))) > 0:
                            return True
                    tmp_src = next_hop

                # Detect confliction with intra-task.
                tmp_src = use_NIC_list[src]
                for next_hop in hop_list:
                    if self_conflict.get((tmp_src, next_hop)):
                        link_num = Simulator.get_infrastructure().get_graph().get_edge_weight_dict()[(tmp_src, next_hop)]
                        self_conflict[(tmp_src, next_hop)] += 1
                        if self_conflict.get((tmp_src, next_hop)) > link_num:
                            return True
                    else:
                        self_conflict[(tmp_src, next_hop)] = 1
                    tmp_src = next_hop

        return False


    def get_butterfly2_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under butterfly strategy.
        [
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        butterfly_pair_list = []
        round_num = math.log2(task_occupied_NIC_num)
        assert(round_num.is_integer())
        round_num = int(round_num)
    
        # Reduce-Scatter
        mask = 1
        communication_size = model_size / 2
        for _ in range(0, round_num):
            a_round = []
            for pair in range(0, task_occupied_NIC_num):
                NIC_src = pair
                NIC_dst = (pair ^ mask)
                a_round.append((NIC_src, NIC_dst, communication_size))
            butterfly_pair_list.append(a_round)
            mask = mask * 2
            communication_size = communication_size / 2

        # All-Gather
        final_butterfly_pair_list = butterfly_pair_list.copy()
        length = len(butterfly_pair_list)
        for i in range(length - 1, -1, -1):
            final_butterfly_pair_list.append(butterfly_pair_list[i])

        return final_butterfly_pair_list
