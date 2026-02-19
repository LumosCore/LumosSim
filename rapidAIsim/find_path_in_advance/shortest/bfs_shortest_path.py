import math
import queue
from collections import defaultdict


class BfsShortestPathStatic:
    OCCUPIED_BEFOREHAND_PATH_RECORD = defaultdict(int)

    LEAF_TO_SPINE_MAP = {}  # {GPUid: spineid}
    LEAF_EGRESS_OCCUPIED = {}  # {leafid: {occupied next_hop_spineid: True, ...}}
    SPINE_TO_LEAF_MAP = {}  # {GPUid: leafid}
    SPINE_EGRESS_OCCUPIED = {}  # {spineid: {occupied next_hop_leafid: True, ...}}

    def __init__(self):
        raise Exception("BfsShortestPathStatic acts as global static class and should not be instantiated!")

    @staticmethod
    def set_path_dict(taskid):
        from rapidAIsim.core.simulator import Simulator
        rapid_graph = Simulator.get_infrastructure().get_graph()

        device_path_dict = Simulator.get_infrastructure().get_device_path_dict()

        print("debug device_path_dict", taskid, len(device_path_dict))

        device_set = rapid_graph.get_vertex_set()
        if (Simulator.CONF_DICT['joint_scheduler'] in ['oxc_scheduler', 'static_scheduler', 'mesh_scheduler',
                                                       'NaiveScheduler', 'NaiveSchedulerCompare', 'OCSExpander',
                                                       'ELEExpander','ELEExpander_SuperPod', 'OCSExpander_superPod','GPUPlacementer', 'GPUPlacementer2',
                                                       'GPUPlacementer3', 'GPUPlacementer4', 'StaticPlacementer',
                                                       'StaticPlacementerRelax', 'StaticPlacementerAI']
                or Simulator.CONF_DICT['task_type'] == 'compare_with_netbench'):
            shortest_path_length = BfsShortestPathStatic.get_shortest_path_length(rapid_graph, device_set)
        else:
            shortest_path_length = BfsShortestPathStatic.get_shortest_path_length_fast(rapid_graph, device_set,
                                                                                       device_path_dict)
        for src in device_set:
            for dst in device_set:
                if src != dst:
                    out_edge_dict = rapid_graph.get_out_edge_dict_from(src)
                    for next_hop_id in out_edge_dict:
                        if shortest_path_length[src][dst] == shortest_path_length[next_hop_id][dst] + 1:
                            device_path_dict[src].add_to_next_hop_dict(dst, next_hop_id, taskid)
        if Simulator.CONF_DICT['joint_scheduler'] in ['oxc_scheduler', 'static_scheduler']:
            BfsShortestPathStatic.add_NIC_spine_map(rapid_graph, device_set)

        if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
            NIC_num = int(Simulator.CONF_DICT['NIC_num'])
            BfsShortestPathStatic.record_beforehand_path(device_path_dict, NIC_num)

        print("finish bfs")

    @staticmethod
    def get_shortest_path_length(graph, device_set):
        # shortest_path_length = [[float("inf") for _ in range(max(device_set) + 1)]
        #                          for _ in range(max(device_set) + 1)]

        shortest_path_length = {}
        for src in device_set:
            shortest_path_length[src] = {}
            for dst in device_set:
                shortest_path_length[src][dst] = float("inf")

        # import time
        # start_time = time.time()
        for device_id in device_set:
            shortest_path_length[device_id][device_id] = 0
            bfs_queue = queue.Queue()
            record_gone_dict = {}
            cur_layer = 0
            bfs_queue.put((device_id, cur_layer))

            while bfs_queue.empty() is False:
                tmp_src, cur_layer = bfs_queue.get()
                record_gone_dict[tmp_src] = True
                connect_list = graph.get_out_edge_dict_from(tmp_src)
                for tmp_dst in connect_list:
                    if tmp_dst not in record_gone_dict:
                        shortest_path_length[tmp_src][tmp_dst] = 1
                        shortest_path_length[device_id][tmp_dst] = cur_layer + 1
                        record_gone_dict[tmp_dst] = True
                        bfs_queue.put((tmp_dst, cur_layer + 1))
        # consuming_time = time.time() - start_time
        # print('Consuming time of calculating ECMP routing:', consuming_time)
        return shortest_path_length

    @staticmethod
    def get_shortest_path_length_fast(graph, device_set, device_path_dict):
        from rapidAIsim.core.simulator import Simulator
        # shortest_path_length = [[float("inf") for _ in range(max(device_set) + 1)]
        #                         for _ in range(max(device_set) + 1)]
        # import time
        # start_time = time.time()

        # A GPU is bound to a leaf switch, so we only need to calculate shortest paths between switches.
        NIC_num = int(Simulator.CONF_DICT['NIC_num'])
        NIC_set = set([i for i in range(NIC_num)])
        switch_set = device_set ^ NIC_set

        NIC_switch_map = {}
        for nic in NIC_set:
            switch_id = device_path_dict[nic].get_connected_to_list()[0]
            NIC_switch_map[nic] = switch_id
            NIC_switch_map[switch_id] = nic

        shortest_path_length = {}
        for src in device_set:
            shortest_path_length[src] = {}
            for dst in device_set:
                if src != dst:
                    shortest_path_length[src][dst] = float("inf")
                else:
                    shortest_path_length[src][dst] = 0

        for device_id in switch_set:
            # shortest_path_length[device_id][device_id] = 0
            bfs_queue = queue.Queue()
            record_gone_dict = {}
            cur_layer = 0
            bfs_queue.put((device_id, cur_layer))

            while bfs_queue.empty() is False:
                tmp_src, cur_layer = bfs_queue.get()
                record_gone_dict[tmp_src] = True
                connect_list = graph.get_out_edge_dict_from(tmp_src)
                for tmp_dst in connect_list:
                    if tmp_dst >= NIC_num:  # If tmp_dst is switch
                        if tmp_dst not in record_gone_dict:
                            shortest_path_length[tmp_src][tmp_dst] = 1
                            shortest_path_length[device_id][tmp_dst] = cur_layer + 1
                            record_gone_dict[tmp_dst] = True
                            bfs_queue.put((tmp_dst, cur_layer + 1))

        for src in NIC_set:
            for dst in device_set:
                if src != dst:
                    bind_switch = NIC_switch_map[src]
                    if dst == bind_switch:
                        shortest_path_length[src][dst] = 1
                    else:
                        shortest_path_length[src][dst] = shortest_path_length[bind_switch][dst] + 1

        for src in device_set:
            for dst in NIC_set:
                if src != dst:
                    bind_switch = NIC_switch_map[dst]
                    if src == bind_switch:
                        shortest_path_length[src][dst] = 1
                    else:
                        shortest_path_length[src][dst] = shortest_path_length[src][bind_switch] + 1

        # consuming_time = time.time() - start_time
        # print('Consuming time of calculating ECMP routing:', consuming_time)
        return shortest_path_length

    @staticmethod
    def add_NIC_spine_map(rapid_graph, device_set):
        from rapidAIsim.core.simulator import Simulator
        gpu_list = []
        leaf_list = []
        spine_list = []
        for dev in device_set:
            if Simulator.is_spine_switch(dev):
                spine_list.append(dev)
            if Simulator.is_leaf_switch(dev):
                leaf_list.append(dev)
            if Simulator.is_GPU(dev):
                gpu_list.append(dev)

        if not spine_list:
            return

        gpu_list.sort()
        num_connected_gpus_per_leaf = []
        # Calculate the number of connected gpus for every leaf block
        for leafid in leaf_list:
            num_connected_gpus_per_leaf.append(rapid_graph.get_vertex_to_vertexset_linknum(leafid, gpu_list))

        # min_leaf_size = min(num_connected_gpus_per_leaf)
        min_leaf_size = BfsShortestPathStatic.multi_gcd(num_connected_gpus_per_leaf)

        num_ports_per_spine = []
        # Calculate the number of ports for each spine
        for spineid in spine_list:
            num_ports_per_spine.append(rapid_graph.get_vertex_to_vertexset_linknum(spineid, leaf_list))

        total_spine_ports = sum(num_ports_per_spine)
        spine_allocation_per_leaf = []
        for i in range(len(spine_list)):
            num_spines = int(num_ports_per_spine[i] * min_leaf_size / total_spine_ports)
            for j in range(num_spines):
                spine_allocation_per_leaf.append(spine_list[i])

        if len(spine_allocation_per_leaf) == min_leaf_size:
            for i in range(len(gpu_list)):
                # Simulator.NIC_SPINE_MAP[gpu_list[i]] = spine_allocation_per_leaf[i % min_leaf_size]
                Simulator.set_NIC_to_spine_map(gpu_list[i], spine_allocation_per_leaf[i % min_leaf_size])
        else:
            for i in range(len(gpu_list)):
                # Simulator.NIC_SPINE_MAP[gpu_list[i]] = -1
                Simulator.set_NIC_to_spine_map(gpu_list[i], -1)
            raise Exception('Subclos is not normal.')

    @staticmethod
    def multi_gcd(data_list):
        from functools import reduce
        return reduce(math.gcd, data_list)

    @staticmethod
    def record_beforehand_path(device_path_dict, NIC_num):
        for src in range(NIC_num):
            for dst in range(NIC_num):
                if src != dst:
                    BfsShortestPathStatic.select_hops(device_path_dict, src, dst)

    @staticmethod
    def add_OCCUPIED_BEFOREHAND_PATH_RECORD(src, dst):
        BfsShortestPathStatic.OCCUPIED_BEFOREHAND_PATH_RECORD[(src, dst)] += 1

    @staticmethod
    def get_OCCUPIED_BEFOREHAND_PATH_RECORD(src, dst):
        return BfsShortestPathStatic.OCCUPIED_BEFOREHAND_PATH_RECORD.get((src, dst))

    @staticmethod
    def select_hops(device_path_dict, src, dst, taskid=-2):
        from rapidAIsim.core.simulator import Simulator

        hop_list = []
        # Find subsequent paths.
        next_hop = None
        tmp_src = src

        while next_hop != dst:
            next_hop_dict = device_path_dict[tmp_src].get_to_next_hop_dict(taskid)

            to_dst_next_hop_list = next_hop_dict[dst]

            if len(to_dst_next_hop_list) == 1:
                next_hop = to_dst_next_hop_list[0]
                hop_list.append(next_hop)
                tmp_src = next_hop
                continue

            for tmp_next_hop in to_dst_next_hop_list:
                # leaf layer -> spine layer routing
                if tmp_next_hop > tmp_src:
                    if BfsShortestPathStatic.LEAF_TO_SPINE_MAP.get(src):
                        hop_list.append(BfsShortestPathStatic.LEAF_TO_SPINE_MAP.get(src))
                        tmp_src = tmp_next_hop
                        next_hop = tmp_next_hop
                        break
                    else:
                        if BfsShortestPathStatic.LEAF_EGRESS_OCCUPIED.get(tmp_src):
                            if tmp_next_hop not in BfsShortestPathStatic.LEAF_EGRESS_OCCUPIED.get(tmp_src):
                                BfsShortestPathStatic.LEAF_EGRESS_OCCUPIED[tmp_src][tmp_next_hop] = True
                                BfsShortestPathStatic.LEAF_TO_SPINE_MAP[src] = tmp_next_hop
                                hop_list.append(tmp_next_hop)
                                tmp_src = tmp_next_hop
                                next_hop = tmp_next_hop
                                break
                        else:
                            BfsShortestPathStatic.LEAF_EGRESS_OCCUPIED[tmp_src] = {}
                            BfsShortestPathStatic.LEAF_EGRESS_OCCUPIED[tmp_src][tmp_next_hop] = True
                            BfsShortestPathStatic.LEAF_TO_SPINE_MAP[src] = tmp_next_hop
                            hop_list.append(tmp_next_hop)
                            tmp_src = tmp_next_hop
                            next_hop = tmp_next_hop
                            break
                # spine layer -> leaf layer routing
                elif tmp_next_hop < tmp_src:
                    if BfsShortestPathStatic.SPINE_TO_LEAF_MAP.get(dst):
                        hop_list.append(BfsShortestPathStatic.SPINE_TO_LEAF_MAP.get(dst))
                        tmp_src = tmp_next_hop
                        next_hop = tmp_next_hop
                        break
                    else:
                        if BfsShortestPathStatic.SPINE_EGRESS_OCCUPIED.get(tmp_src):
                            if tmp_next_hop not in BfsShortestPathStatic.SPINE_EGRESS_OCCUPIED.get(tmp_src):
                                BfsShortestPathStatic.SPINE_EGRESS_OCCUPIED[tmp_src][tmp_next_hop] = True
                                BfsShortestPathStatic.SPINE_TO_LEAF_MAP[dst] = tmp_next_hop
                                hop_list.append(tmp_next_hop)
                                tmp_src = tmp_next_hop
                                next_hop = tmp_next_hop
                                break
                        else:
                            BfsShortestPathStatic.SPINE_EGRESS_OCCUPIED[tmp_src] = {}
                            BfsShortestPathStatic.SPINE_EGRESS_OCCUPIED[tmp_src][tmp_next_hop] = True
                            BfsShortestPathStatic.SPINE_TO_LEAF_MAP[dst] = tmp_next_hop
                            hop_list.append(tmp_next_hop)
                            tmp_src = tmp_next_hop
                            next_hop = tmp_next_hop
                            break
                else:
                    raise Exception("tmp_next_hop cannot be equal to tmp_src!")
            else:
                raise Exception("next_hop cannot be equal to tmp_src!")

        Simulator.BEFOREHAND_PATH[(src, dst)] = hop_list
