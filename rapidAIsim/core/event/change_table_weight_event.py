from .event import Event
from rapidAIsim.utils.traffic_matrix_helper import get_traffic_matrix
import numpy as np


class ChangeTableWeightEvent(Event):
    def __init__(self, relative_time_from_now, time_interval):
        """
        :param time_interval 表示测量流量矩阵的时间间隔。
        """
        super().__init__(relative_time_from_now)
        self.time_interval = time_interval
        self._type_priority = 4

    def do_sth(self):
        from rapidAIsim.core.simulator import Simulator
        hist_len = int(Simulator.CONF_DICT['figret_hist_len'])
        traffic_matrix = get_traffic_matrix(self.event_time)
        Simulator.history_traffic_matrix.append(traffic_matrix)
        if len(Simulator.history_traffic_matrix) == hist_len:
            traffic_matrix = np.array(Simulator.history_traffic_matrix)
            Simulator.history_traffic_matrix.pop(0)
            # change_weight
            self.calculate_and_change_weight(traffic_matrix)
        # register next event
        if len(Simulator._event_q) != 0:
            # avoid registering the event when the simulator is shutting down
            Simulator.register_event(ChangeTableWeightEvent(self.time_interval, self.time_interval))

    def calculate_and_change_weight(self, traffic_matrices):
        """
        调用Figret的推理模型，计算每条路径的分流比例，然后根据分流比例调整每条路径的权重。
        """
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        num_nodes = infra.pod_num + infra.spine_switch_num
        spine_start_index = infra.NIC_num + infra.leaf_switch_num
        split_ratios, split_ratios_ori = Simulator.figret.inference(traffic_matrices, infra.spine_switch_num)
        if Simulator.split_ratios is None:
            Simulator.split_ratios = split_ratios_ori
        else:
            # 在线训练，更新模型
            Simulator.figret.get_single_loss_and_backword(split_ratios_ori, traffic_matrices, infra.spine_switch_num)
            Simulator.split_ratios = split_ratios_ori
        direct_table = Simulator.inter_pod_weighted_direct_table
        twohop_table = Simulator.inter_pod_weighted_twohop_table
        iter_ratios = iter(split_ratios)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                paths = Simulator.figret.env.pij[(i, j)]
                ratios = next(iter_ratios)
                # if len(paths)>0:
                #     print("debug path")
                #     print(paths)
                #     print("debug ratios")
                #     print(ratios)
                assert len(paths) == len(ratios), \
                    f"path length does not match: {len(paths)} != {len(ratios)}"
                for new_weight, edge_path in zip(ratios, paths):
                    path = []
                    for s, d in edge_path:
                        path.append(s)
                    path.append(edge_path[-1][-1])
                    dst_pod = path[-1]
                    src_spine = path[1] - infra.pod_num + spine_start_index
                    next_hop_spine = path[2] - infra.pod_num + spine_start_index

                    if len(path) == 4:
                        for k, (next_hop, weight) in enumerate(direct_table[src_spine][dst_pod]):
                            if next_hop == next_hop_spine:
                                direct_table[src_spine][dst_pod][k] = (next_hop_spine, float(new_weight))
                    else:
                        for k, (next_hop, weight) in enumerate(twohop_table[src_spine][dst_pod]):
                            if next_hop == next_hop_spine:
                                twohop_table[src_spine][dst_pod][k] = (next_hop_spine, float(new_weight))
