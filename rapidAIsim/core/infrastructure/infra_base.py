from typing import Dict, Tuple, Set, DefaultDict
from collections import defaultdict
from rapidAIsim.core import Flow
from rapidAIsim.core.infrastructure.device import Device
from rapidAIsim.core.infrastructure.graph import Graph
from rapidAIsim.find_path_in_advance.shortest.bfs_shortest_path import BfsShortestPathStatic


class InfraBase:
    """Create network infrastructure according to selectors and configs.
    """

    def __init__(self) -> None:
        self._graph = None
        self._link_capacity_dict: Dict[int, Dict[Tuple[str, str], float]] = {}
        self._link_flow_occupy_dict: DefaultDict[int, DefaultDict[Tuple[str, str], Set[int]]] = \
            defaultdict(lambda: defaultdict(set))
        self._flow_infly_info_dict: Dict[int, Flow] = {}
        self._device_path_dict = {}  # {device_id: Device} records the next hop when flow arrives every device.
        self._used_gpu_state_map = {}
        self._job_gpu_list_map = {}
        self._history_flow_info_dict = {}

        from rapidAIsim.core.simulator import Simulator
        config = Simulator.CONF_DICT
        self.topo_type = config['topo_type']
        self.layers = int(config['layers'])
        self.NIC_num = int(config['NIC_num'])
        self.pod_num = int(config['pod_num'])
        self.leaf_switch_num = int(config['leaf_switch_num'])
        self.leaf_switch_port_num = int(config['leaf_switch_port_num'])
        self.spine_switch_num = int(config['spine_switch_num'])
        self.spine_switch_port_num = int(config['spine_switch_port_num'])
        self.spine_up_port_num = self.spine_switch_port_num // 2
        self.server_num_per_pod = int(config['server_num_per_pod'])
        self.NIC_num_in_a_server = int(config['NIC_num_in_a_server'])
        self.spine_num_per_pod = self.spine_switch_num // self.pod_num
        self.leaf_num_per_pod = self.leaf_switch_num // self.pod_num
        self.server_num = self.NIC_num // self.NIC_num_in_a_server
        if self.server_num_per_pod != self.server_num // self.pod_num:
            raise ValueError("server_num_per_pod should be equal to server_num // pod_num, but got {} and {}"
                             .format(self.server_num_per_pod, self.server_num // self.pod_num))
        self.server_per_leaf = self.server_num // self.leaf_switch_num

        try:
            self.ocs_num = int(config['ocs_num'])
        except (KeyError, ValueError):
            self.ocs_num = 0
        try:
            self.leaf_spine_link_num = int(config['leaf_spine_link_num'])
        except (KeyError, ValueError):
            self.leaf_spine_link_num = 1

    def create_topology(self, connect_info_list):

        self._graph = Graph(connect_info_list)

        self.set_init_link_capacity_and_occupied()

        vertex_set = self._graph.get_vertex_set()
        for index in vertex_set:
            self._device_path_dict[index] = Device(index)

        for (start_node_index, end_node_index, link_num) in connect_info_list:
            self.create_connection(start_node_index, end_node_index, link_num)

    def create_connection(self, src, dst, link_num):
        src_device = self._device_path_dict[src]
        src_device.add_connect(dst, link_num)

    def find_all_path(self, task_id):
        from rapidAIsim.core.simulator import Simulator
        find_path_method = Simulator.CONF_DICT['find_path_method']
        if find_path_method == 'shortest':
            BfsShortestPathStatic.set_path_dict(task_id)
        # if find_path_method == 'updown' and task_id == 0:
        #     BfsShortestPathStatic.set_path_dict(task_id)

    def refresh_link_flow_occupy_dict(self):
        self._link_flow_occupy_dict = defaultdict(lambda: defaultdict(set))

    def update_flow_route_info(self, task_id):
        flow_infly_info_dict = self.get_flow_infly_info_dict()
        for flow_id, flow in flow_infly_info_dict.items():
            if flow.get_task_id() == task_id:
                flow.find_hop_list()
                src = flow.get_src()
                flow_id = flow.get_flow_id()
                hop_list = flow.get_hop_list()

                # Start subsequent paths.
                tmp_src = src
                for next_hop in hop_list:
                    self.add_link_flow_occupy(flow_id, tmp_src, next_hop, task_id)
                    tmp_src = next_hop
                self.set_flow_infly_info(flow_id, flow, task_id)  # Necessary

    def reconfigure_topo(self, connect_info_list, task_id):
        """Reconfigure topology
        """
        self._graph.refresh_graph_connection_info(connect_info_list)

        # Reconfiguration according to task_id
        self.set_link_capacity(task_id)

        vertex_set = self._graph.get_vertex_set()
        for index in vertex_set:
            if not self._device_path_dict.get(index):
                self._device_path_dict[index] = Device(index)
            else:
                self._device_path_dict[index].clear_to_next_hop_dict(task_id)

        for (start_node_index, end_node_index, link_num) in connect_info_list:
            self.create_connection(start_node_index, end_node_index, link_num)

    def get_device_path_dict(self):
        """Return device_path_dict which records the next hop when flow arrives every device.
        """
        return self._device_path_dict

    def get_device(self, device_id):
        return self._device_path_dict[device_id]

    def get_graph(self):
        return self._graph

    def set_init_link_capacity_and_occupied(self):
        from rapidAIsim.core.simulator import Simulator
        switch_port_bandwidth = float(Simulator.CONF_DICT['switch_port_bandwidth'])
        edge_weight_dict = self._graph.get_edge_weight_dict()
        self._link_capacity_dict[-2] = {}
        self._link_flow_occupy_dict[-2] = defaultdict(set)
        for (src, dst), link_num in edge_weight_dict.items():
            # -2 means no reconfiguration
            real_link_num = link_num
            from rapidAIsim.core.simulator import Simulator
            if (src,dst) in Simulator.rerouting_link:
                assert link_num >= Simulator.rerouting_link[(src,dst)]
                real_link_num -= Simulator.rerouting_link[(src,dst)]*3/4
                # real_link_num = 0
            self._link_capacity_dict[-2][(src, dst)] = real_link_num * switch_port_bandwidth
            self._link_flow_occupy_dict[-2][(src, dst)] = set()

    def set_link_capacity(self, task_id):
        from rapidAIsim.core.simulator import Simulator
        switch_port_bandwidth = float(Simulator.CONF_DICT['switch_port_bandwidth'])
        edge_weight_dict = self._graph.get_edge_weight_dict()

        if Simulator.CONF_DICT['joint_scheduler'] in ['OCSExpander', 'OCSExpander_superPod','ELEExpander','ELEExpander_SuperPod']:
            # # -2 means no reconfiguration
            task_id = -2

        if task_id not in self._link_capacity_dict:
            self._link_capacity_dict[task_id] = {}

        for (src, dst), link_num in edge_weight_dict.items():
            # src_type = Simulator.get_device_type(src)
            # dst_type = Simulator.get_device_type(dst)
            # print("debug: link_num: {}-{}, {}-{}, {}".format(src, dst, src_type, dst_type, link_num))
            real_link_num = link_num
            from rapidAIsim.core.simulator import Simulator
            if (src,dst) in Simulator.rerouting_link:
                assert link_num >= Simulator.rerouting_link[(src,dst)]
                real_link_num -= Simulator.rerouting_link[(src,dst)]*3/4
                # real_link_num = 0
                print("debug rerouting",src,dst,real_link_num,link_num)
                # print(Simulator.rerouting_link)
                # assert False
            # print("debug rerouting",src,dst,real_link_num,link_num)
            self._link_capacity_dict[task_id][(src, dst)] = real_link_num * switch_port_bandwidth
        # assert False

    def get_link_capacity_dict(self, task_id):
        if task_id not in self._link_capacity_dict:
            # -2 means no reconfiguration
            return self._link_capacity_dict[-2]
        return self._link_capacity_dict[task_id]

    def get_a_link_capacity(self, src, dst, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.is_GPU(src) and Simulator.is_GPU(dst) and Simulator.is_in_the_same_server(src, dst):
            return float(Simulator.CONF_DICT['inner_server_bandwidth'])

        if task_id not in self._link_capacity_dict:
            # -2 means no reconfiguration
            if (src, dst) not in self._link_capacity_dict[-2]:
                print("Error: No such link in link_capacity_dict[-2]")
                print(self._link_capacity_dict[-2])
            return self._link_capacity_dict[-2][(src, dst)]
        return self._link_capacity_dict[task_id][(src, dst)]

    def add_link_flow_occupy(self, flow_id, src, dst, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['reconfiguration'] == 'no':
            # -2 means no reconfiguration
            task_id = -2
        self._link_flow_occupy_dict[task_id][(src, dst)].add(flow_id)

    def del_link_flow(self, flow_id, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['reconfiguration'] == 'no':
            task_id = -2
        for flows in self._link_flow_occupy_dict[task_id].values():
            if flow_id in flows:
                flows.remove(flow_id)

    def get_link_num(self, src, dst, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['reconfiguration'] == 'no':
            task_id = -2
            if task_id not in self._link_flow_occupy_dict:
                print('Now no flows.')
                return 0
            return len(self._link_flow_occupy_dict[task_id][(src, dst)])
        else:
            if (task_id in self._link_flow_occupy_dict and (src, dst) in self._link_flow_occupy_dict[task_id]
                    and dst <= int(Simulator.CONF_DICT['NIC_num'])
                    and task_id in Simulator.task_class
                    and Simulator.task_class[task_id] == 'alltoallv'):
                return len(self._link_flow_occupy_dict[task_id][(src, dst)])
            return 0

    def del_link_flow_occupy(self, flow_id, src, dst, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['reconfiguration'] == 'no':
            # # -2 means no reconfiguration
            task_id = -2
        self._link_flow_occupy_dict[task_id][(src, dst)].remove(flow_id)

    def get_link_flow_occupy_dict(self):
        return self._link_flow_occupy_dict

    def get_link_flow_occupy_dict_given_task_id(self, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['reconfiguration'] == 'no':
            # -2 means no reconfiguration
            task_id = -2
        return self._link_flow_occupy_dict[task_id]

    def get_link_flow_occupy_list(self, src, dst, task_id):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['reconfiguration'] == 'no':
            # -2 means no reconfiguration
            task_id = -2
        if task_id not in self._link_flow_occupy_dict:
            print('Now no flows.')
            return []
        return sorted(list(self._link_flow_occupy_dict[task_id][(src, dst)]))

    def set_flow_infly_info(self, flow_id, flow, task_id):
        hop_list = flow.get_hop_list()
        src = flow.get_src()
        tmp_src = src
        # Affect other flows capacity
        for next_hop in hop_list:
            link_flow_occupy_list = self.get_link_flow_occupy_list(tmp_src, next_hop, task_id)
            for affected_flow_id in link_flow_occupy_list:
                if affected_flow_id != flow_id:
                    affected_flow = self.get_flow_from_infly_info_dict(affected_flow_id)
                    affected_flow.set_expected_finish_time(-1)
                    affected_flow.set_need_recalculation(True)
            tmp_src = next_hop

        self._flow_infly_info_dict[flow_id] = flow

    def del_a_flow_infly_info(self, flow_id, task_id):
        flow = self._flow_infly_info_dict[flow_id]
        hop_list = flow.get_hop_list()
        src = flow.get_src()
        tmp_src = src
        # Affect other flows capacity
        for next_hop in hop_list:
            link_flow_occupy_list = self.get_link_flow_occupy_list(tmp_src, next_hop, task_id)
            for affected_flow_id in link_flow_occupy_list:
                if affected_flow_id != flow_id:
                    affected_flow = self.get_flow_from_infly_info_dict(affected_flow_id)
                    affected_flow.set_expected_finish_time(-1)
                    affected_flow.set_need_recalculation(True)
            tmp_src = next_hop

        del self._flow_infly_info_dict[flow_id]

    def del_flows_infly_info(self, flow_ids, task_id):
        affected_flows = set()
        for flow_id in flow_ids:
            flow = self._flow_infly_info_dict[flow_id]
            hop_list = flow.get_hop_list()
            src = flow.get_src()
            tmp_src = src
            # Affect other flows capacity
            for next_hop in hop_list:
                link_flow_occupy_list = self.get_link_flow_occupy_list(tmp_src, next_hop, task_id)
                for affected_flow_id in link_flow_occupy_list:
                    if affected_flow_id not in flow_ids and affected_flow_id not in affected_flows:
                        affected_flows.add(affected_flow_id)
                tmp_src = next_hop
            del self._flow_infly_info_dict[flow_id]
        for affected_flow_id in affected_flows:
            affected_flow = self.get_flow_from_infly_info_dict(affected_flow_id)
            affected_flow.set_expected_finish_time(-1)
            affected_flow.set_need_recalculation(True)

    def get_flow_infly_info_dict(self):
        return self._flow_infly_info_dict

    def get_history_flow_info_dict(self):
        return self._history_flow_info_dict

    def get_flow_from_infly_info_dict(self, flow_id):
        if flow_id not in self._flow_infly_info_dict:
            for key in self._flow_infly_info_dict:
                print("debug try to get", flow_id, key)
        return self._flow_infly_info_dict[flow_id]
