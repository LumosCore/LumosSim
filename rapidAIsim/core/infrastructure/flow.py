import random
import warnings

import mmh3
from copy import deepcopy
import time
import math


class Flow:
    def __init__(self, flow_id, size, start_time, src, dst, remainder_size, last_calculated_time, taskid, round_id,
                 task_occupied_NIC_num, conservative=False, _total_round=1,
                 need_prior_calculate=True, need_subsequent_calculate=False) -> None:
        """A flow is transmitted over the network.
        Args:
            - flow_id: A flow identification.
            - size: Total size.
            - start_time: Original start time.
            - src: The original source.
            - dst: The final destination.
            - remainder_size: remainder size waiting to be transmitted.
            - last_calculated_time: Record last calculated time.
            - taskid: The flow belongs to taskid.
            - round_id: The flow belongs to round_id in taskid.
            - need_prior_calculate: whether need prior calculation.
            - need_subsequent_calculate: whether need subsequent calculation.
        """
        self._flow_id = flow_id
        self._size = size
        self._start_time = start_time
        self._src = src
        self._dst = dst
        self._remainder_size = remainder_size
        self._last_calculated_time = last_calculated_time
        self._taskid = taskid
        self._round_id = round_id
        self._task_occupied_NIC_num = task_occupied_NIC_num
        self._conservative = conservative
        self._total_round = _total_round
        self._need_recalculation = True
        self._need_prior_calculate = need_prior_calculate
        self._need_subsequent_calculate = need_subsequent_calculate
        self.sync_delay_alpha = 0

        # The following 4 attributes are used for traffic matrix.
        self._last_statistic_time = -1  # 上一次该流被统计进流量矩阵的时间
        self._traffic_matrix_remainder_size = size  # 该流在上一次统计后剩余的流量

        if conservative is True and src != dst:
            from rapidAIsim.core.simulator import Simulator
            hop_list = Simulator.get_hop_list_from_beforehand_path(src, dst)
            self.set_hop_list(hop_list)
        else:
            self._hop_list = []

        self._min_available_capacity = -1
        self._expected_finish_time = -1

        self._in_the_same_server_flag = False

        self._inter_region_bandwidth = None
        self._inter_node_bandwidth = None

        self._alpha = 0
        from rapidAIsim.core.simulator import Simulator
        self._cwnd = float(Simulator.CONF_DICT['switch_port_bandwidth'])
        self.has_been_transited = False

    def __repr__(self) -> str:
        print_dict = {
            'flow_id': self._flow_id,
            'size': self._size,
            'start_time': self._start_time,
            'src': self._src,
            'dst': self._dst,
            'remainder_size': self._remainder_size,
            'last_calculated_time': self._last_calculated_time,
            'hop_list': self._hop_list,
            'taskid': self._taskid,
            'round_id': self._round_id,
            'min_available_capacity': self._min_available_capacity,
            'expected_finish_time': self._expected_finish_time,
            'in_the_same_server_flag': self._in_the_same_server_flag,
            'task_occupied_NIC_num': self._task_occupied_NIC_num,
        }
        print_str = '<Flow | '
        for key, val in print_dict.items():
            print_str += key + ': ' + str(val) + ', '
        print_str += '>'
        return print_str

    def __lt__(self, other):
        from rapidAIsim.core.simulator import Simulator
        if self._expected_finish_time == -1:
            print('debug raise flow', self, Simulator.get_current_time())
            raise Exception('flow._expected_finish_time cannot be -1 when using heapify')
        elif self._expected_finish_time < other.get_expected_finish_time():
            return True
        else:
            return False

    def find_hop_list(self, need_routing_seed=0, need_rerouting=False):
        from rapidAIsim.core.simulator import Simulator
        # Deal with GPUs in the same server.
        if Simulator.is_in_the_same_server(self._src, self._dst):
            self.set_in_the_same_server()
            return

        # conservative method does not use static routing.
        if self._conservative is True:
            return

        hop_list = []
        # Deal with only 1 GPU occupation
        if self._dst == self._src:
            hop_list = [self._dst]
            self._hop_list = hop_list
            return hop_list

        if Simulator.CONF_DICT['joint_scheduler'] == 'mesh_scheduler':
            hop_list = self._get_mesh_apx_clos_hop_list(self._src, self._dst)
            self._hop_list = hop_list
            return hop_list

        if Simulator.CONF_DICT['joint_scheduler'] == 'mesh_cross':
            hop_list = self._get_mesh_cross_hop_list(self._src, self._dst)
            self._hop_list = hop_list
            return hop_list

        if Simulator.CONF_DICT['joint_scheduler'] in ['hw_oxc_all2all', 'hw_oxc_all2all_sz', 'hw_oxc_all2all2',
                                                      'hw_oxc_allreduce', 'hw_oxc_hdallreduce']:
            hop_list = self._get_hw_oxc_all2all_hop_list(self._src, self._dst)
            self._hop_list = hop_list
            return hop_list

        if Simulator.CONF_DICT['joint_scheduler'] in ['ELEExpander','ELEExpander_SuperPod']:
            base_time = time.time()
            self.sync_delay_alpha = 0
            sync_delay_alpha = float(Simulator.CONF_DICT['sync_delay_alpha'])
            if Simulator.CONF_DICT['find_next_hop_method'] == 'balance':
                hop_list = self._get_up_down_routing_balance(self._src, self._dst)
            else:
                rehashing_time = int(Simulator.CONF_DICT['max_rehashing_time'])
                hop_list = self._get_up_down_routing_new(self._src, self._dst, self._taskid, rehashing_time+round(random.uniform(0, 0.7)))
            self._hop_list = hop_list
            # print("debug end up down cost", time.time() -base_time )
            # print("debug hop_list")
            # print(self._src, self._dst,self._flow_id,self._size)
            # self.sync_delay_alpha = 4 * sync_delay_alpha
            return hop_list

        if Simulator.CONF_DICT['joint_scheduler'] in ['OCSExpander_superPod','OCSExpander']:
            # base_time = time.time()
            # print("debug start up down")
            self.sync_delay_alpha = 0
            sync_delay_alpha = float(Simulator.CONF_DICT['sync_delay_alpha'])
            if Simulator.CONF_DICT['ocs_reconfiguration'] == 'yes':
                find_next_hop_method = Simulator.CONF_DICT['find_next_hop_method']
                if find_next_hop_method == 'balance':
                    hop_list = self._get_up_down_routing_OCS_balance(self._src, self._dst)
                elif find_next_hop_method == 'ecmp':
                    rehashing_time = int(Simulator.CONF_DICT['max_rehashing_time'])
                    hop_list = self._get_up_down_routing_OCS_ECMP(self._src, self._dst, self._taskid, rehashing_time)
                else:
                    raise ValueError("The find_next_hop_method is invalid: {}".format(find_next_hop_method))
            else:
                rehashing_time = int(Simulator.CONF_DICT['max_rehashing_time'])
                if Simulator.CONF_DICT['figret_integration'] == 'yes':
                    rehashing_time = 1
                if Simulator.CONF_DICT['rail_optimized'] == 'yes':
                    hop_list = self._get_up_down_routing_OCS_wECMP_rail_optimized(self._src, self._dst, self._taskid, rehashing_time)
                else:
                    hop_list = self._get_up_down_routing_OCS_wECMP(self._src, self._dst, self._taskid, rehashing_time)
            self._hop_list = hop_list
            # print("debug hop_list")
            # print(self._src, self._dst,self._flow_id,self._size)
            # print("debug end up down cost", time.time() -base_time )
            # self.sync_delay_alpha = 3 * sync_delay_alpha
            return hop_list

        infra = Simulator.get_infrastructure()
        # device_path_dict records the next hop after set path dict.
        device_path_dict = infra.get_device_path_dict()

        # Find subsequent paths.
        next_hop = None
        tmp_src = self._src

        find_next_hop_method = Simulator.CONF_DICT['find_next_hop_method']
        assert find_next_hop_method in ['random', 'balance', 'conservative', 'static_routing', 'ecmp', 'wcmp']

        while next_hop != self._dst:
            next_hop_dict = device_path_dict[tmp_src].get_to_next_hop_dict(self._taskid)
            # from rapidAIsim.core.simulator import Simulator
            # find_path_method = Simulator.CONF_DICT['find_path_method']
            # if find_path_method == 'updown':
            #     next_hop_dict = device_path_dict[tmp_src].get_to_next_hop_dict(0)
            to_dst_next_hop_list = next_hop_dict[self._dst]
            
            to_dst_next_hop_list = list(set(to_dst_next_hop_list))

            next_hop = -1
            if (Simulator.is_leaf_switch(tmp_src) and
                    device_path_dict[tmp_src].has_connect(self._src) and
                    (not device_path_dict[tmp_src].has_connect(self._dst)) and
                    Simulator.CONF_DICT['joint_scheduler'] in ['oxc_scheduler', 'static_scheduler']
            ):
                # Source based path dict
                next_hop = device_path_dict[self._src].get_to_spine_id()
            if next_hop == -1:
                # Destination based path dict
                if find_next_hop_method == 'random' and need_rerouting is False:
                    rehashing_time = Simulator.CONF_DICT['max_rehashing_time']
                    rehashing_time = int(rehashing_time) if rehashing_time != '' else 0
                    next_hop = self._get_random_path(tmp_src, to_dst_next_hop_list, hop_list, need_routing_seed,
                                                     rehashing_time, self._src, self._dst)
                elif find_next_hop_method == 'balance' or need_rerouting is True:
                    next_hop = self._get_balance_path(tmp_src, to_dst_next_hop_list, hop_list)
                elif find_next_hop_method == 'static_routing':
                    next_hop = self._get_static_routing_path(tmp_src, to_dst_next_hop_list, hop_list, self._src)

            hop_list.append(next_hop)

            # Update next-hop tmp_src
            tmp_src = next_hop
        self._hop_list = hop_list
        return hop_list

    def _get_mesh_apx_clos_hop_list(self, src, dst):
        from rapidAIsim.core.simulator import Simulator
        downlinks = int(Simulator.CONF_DICT['downlinks'])

        # infra = Simulator.get_infrastructure()

        src_switch = Simulator.get_scheduler().belong_which_leaf_switch(src)
        dst_switch = Simulator.get_scheduler().belong_which_leaf_switch(dst)
        if src_switch == dst_switch:
            return [src_switch, dst]

        switch_list = Simulator.TASK_SWITCH_DICT[self._taskid]
        if len(switch_list) == 2:
            return [src_switch, dst_switch, dst]

        virtual_clos_leaf_spine_link_num = downlinks / (self._task_occupied_NIC_num / downlinks)

        switch_group_map = {}
        cnt = 0
        for switch_id in switch_list:
            switch_group_map[switch_id] = cnt
            cnt += 1

        src_switch_group = switch_group_map[src_switch]
        src_group = src % downlinks // virtual_clos_leaf_spine_link_num
        dst_switch_group = switch_group_map[dst_switch]
        # dst_group = dst % downlinks // virtual_clos_leaf_spine_link_num

        if src_group == src_switch_group:
            return [src_switch, dst_switch, dst]
        else:
            if src_group == dst_switch_group:
                return [src_switch, dst_switch, dst]
            else:
                for mid_switch, mid_switch_group in switch_group_map.items():
                    if mid_switch_group == src_group:
                        return [src_switch, mid_switch, dst_switch, dst]

    def _get_mesh_cross_hop_list(self, src, dst):
        from rapidAIsim.core.simulator import Simulator
        src_switch = Simulator.get_scheduler().belong_which_leaf_switch(src)
        dst_switch = Simulator.get_scheduler().belong_which_leaf_switch(dst)
        if src_switch == dst_switch:
            return [src_switch, dst]
        else:
            return [src_switch, dst_switch, dst]

    def _get_hw_oxc_all2all_hop_list(self, src, dst):
        from rapidAIsim.core.simulator import Simulator
        NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        src_port_serial = src % NIC_num_in_a_server
        src_belong = src // NIC_num_in_a_server
        dst_port_serial = dst % NIC_num_in_a_server
        dst_belong = dst // NIC_num_in_a_server

        if src_belong == dst_belong:
            return [dst]
        else:
            if src_port_serial == dst_belong:
                dst_mid = NIC_num_in_a_server * dst_belong + src_belong
                return [dst_mid, dst]
            src_mid = src_belong * NIC_num_in_a_server + dst_belong
            if dst_port_serial == src_belong:
                return [src_mid, dst]
            else:
                dst_mid = dst_belong * NIC_num_in_a_server + src_belong
                return [src_mid, dst_mid, dst]

    def _get_up_down_routing(self, src, dst, task_seed, rehashing_time=5):
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        random.seed(task_seed + src + dst)
        best_hop_list = []
        best_capacity = 0
        hash_factor = 0
        while hash_factor < rehashing_time:
            hop_list = []
            cur_node = src
            cur_capacity = 1000000
            flag = True
            while cur_node != dst:
                cur_layer = math.ceil(max(0, cur_node - int(Simulator.CONF_DICT['NIC_num'])) / int(
                    Simulator.CONF_DICT['spine_switch_num']))
                if cur_layer > 3:
                    rehashing_time = 0
                    flag = False
                if dst not in Simulator.clos_down_table[cur_node]:
                    next_hop = self._get_random_path(cur_node, Simulator.clos_up_table[cur_node], hop_list,
                                                     hash_factor, rehashing_time, src, dst, flag)
                    hop_list.append(next_hop)

                else:
                    # cur_node = random.choice(Simulator.clos_down_table[cur_node][dst])
                    next_hop = self._get_random_path(cur_node, Simulator.clos_down_table[cur_node][dst],
                                                     hop_list, hash_factor, rehashing_time, src, dst, flag)
                    hop_list.append(next_hop)
                the_link_flow_occupy_num = len(infra.get_link_flow_occupy_list(cur_node, next_hop, task_seed))
                the_link_capacity = infra.get_a_link_capacity(cur_node, next_hop, task_seed)
                if the_link_flow_occupy_num > 0:
                    tmp_link_capacity = the_link_capacity / the_link_flow_occupy_num
                else:
                    tmp_link_capacity = the_link_capacity
                if tmp_link_capacity < cur_capacity:
                    cur_capacity = tmp_link_capacity
                cur_node = next_hop
            if cur_capacity > best_capacity:
                best_capacity = cur_capacity
                best_hop_list = deepcopy(hop_list)
            hash_factor += 1
        return best_hop_list

    def _get_up_down_routing_balance(self, src, dst):
        from rapidAIsim.core.simulator import Simulator
        hop_list = []
        cur_node = src
        while cur_node != dst:
            if dst not in Simulator.clos_down_table[cur_node]:
                next_hop_list = deepcopy(Simulator.clos_up_table[cur_node])
                if Simulator.is_leaf_switch(cur_node):
                    next_hop_list = list(reversed(next_hop_list))
                else:
                    next_hop_list.sort()
                next_hop_list = list(set(next_hop_list))
                next_hop = self._get_balance_path(cur_node, next_hop_list, hop_list)
                # print("debug up balance at time",Simulator.get_current_time())
                # print(f'{src},{dst},{cur_node},{next_hop}')
                # print(next_hop_list)
                hop_list.append(next_hop)
            else:
                next_hop_list = deepcopy(Simulator.clos_down_table[cur_node][dst])
                if Simulator.is_leaf_switch(cur_node):
                    next_hop_list = list(reversed(next_hop_list))
                else:
                    next_hop_list.sort()
                next_hop_list = list(set(next_hop_list))
                next_hop = self._get_balance_path(cur_node, next_hop_list, hop_list)
                # print("debug down balance at time",Simulator.get_current_time())
                # print(f'{src},{dst},{cur_node},{next_hop}')
                # print(next_hop_list)
                hop_list.append(next_hop)
            cur_node = next_hop
        return hop_list

    def _get_up_down_routing_new(self, src, dst, task_seed, rehashing_time=5):
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        random.seed(task_seed + src + dst)
        best_hop_list = []
        best_capacity = 0
        port_routing = Simulator.CONF_DICT['port_routing'] == 'yes'
        for hash_factor in range(rehashing_time):
            hop_list = []
            cur_node = src
            cur_capacity = 1000000
            while cur_node != dst:
                cur_layer = math.ceil(max(0, cur_node - int(Simulator.CONF_DICT['NIC_num'])) / int(
                    Simulator.CONF_DICT['spine_switch_num']))

                if dst not in Simulator.clos_down_table[cur_node]:
                    next_hop_list = Simulator.clos_up_table[cur_node]
                    next_hop_list.sort()
                    if len(next_hop_list) == 1:
                        next_hop = next_hop_list[0]
                    else:
                        next_hop = self._get_random_path(cur_node, next_hop_list, hop_list, hash_factor,
                                                         rehashing_time, src, dst, port_routing)
                    hop_list.append(next_hop)
                    # print("debug updown1",next_hop,next_hop_list,rehashing_time)
                else:
                    next_hop_list = Simulator.clos_down_table[cur_node][dst]
                    next_hop_list.sort()
                    if len(next_hop_list) == 1:
                        next_hop = next_hop_list[0]
                    else:
                        next_hop = self._get_random_path(cur_node, next_hop_list, hop_list, hash_factor,
                                                         rehashing_time, src, dst, port_routing)
                    hop_list.append(next_hop)
                    # print("debug updown2",next_hop,next_hop_list,rehashing_time)
                link_flow_occupy_num = len(infra.get_link_flow_occupy_list(cur_node, next_hop, task_seed))
                link_capacity = infra.get_a_link_capacity(cur_node, next_hop, task_seed)
                tmp_link_capacity = link_capacity / link_flow_occupy_num if link_flow_occupy_num > 0 else link_capacity
                if tmp_link_capacity < cur_capacity:
                    cur_capacity = tmp_link_capacity
                cur_node = next_hop
            if cur_capacity > best_capacity:
                best_capacity = cur_capacity
                best_hop_list = deepcopy(hop_list)
        return best_hop_list

    def _get_up_down_routing_OCS(self, src, dst, taskid, rehashing_time=5):
        """Lumoscore的原始路由算法。暂时弃用。"""
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        best_hop_list = []
        best_capacity = 0
        if 'is_two_iter' in Simulator.CONF_DICT and Simulator.CONF_DICT['is_two_iter'] == 'yes':
            rehashing_time = 10
        port_routing = Simulator.CONF_DICT['port_routing'] == 'yes'
        for hash_factor in range(rehashing_time):
            hop_list = []
            cur_node = src
            cur_capacity = 1000000
            while cur_node != dst:
                if dst not in Simulator.clos_down_table[cur_node]:
                    next_hop_list = Simulator.clos_up_table[cur_node]
                    next_hop_list.sort()
                    next_hop = self._get_random_path(cur_node, next_hop_list, hop_list, hash_factor,
                                                     min(2, rehashing_time), src, dst, port_routing)
                    hop_list.append(next_hop)
                else:
                    pod_gpu_size = int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num'])
                    pod_spine_size = int(Simulator.CONF_DICT['spine_switch_num']) // int(Simulator.CONF_DICT['pod_num'])
                    original_next_hop_list = deepcopy(Simulator.clos_down_table[cur_node][dst])
                    base_spine_id = int(Simulator.CONF_DICT['NIC_num']) + int(Simulator.CONF_DICT['leaf_switch_num'])
                    pod_of_curr_spine = (cur_node - base_spine_id) // pod_spine_size
                    if pod_of_curr_spine == src // pod_gpu_size:
                        # If the current spine switch is in the same pod as the source server
                        next_hop_list = [val for val in original_next_hop_list
                                         if (val - base_spine_id) // pod_spine_size == dst // pod_gpu_size]
                        assert len(next_hop_list) > 0, \
                            "Routing error! No available spine switch to dst pod: from {} to {}".format(
                                pod_of_curr_spine, dst // pod_gpu_size)
                    elif pod_of_curr_spine == dst // pod_gpu_size:
                        # If the current spine switch is in the same pod as the destination server
                        next_hop_list = [val for val in original_next_hop_list if val < base_spine_id]
                        assert len(next_hop_list) > 0, \
                            "Routing error! No available leaf switch to dst server: from {} to {}".format(
                                pod_of_curr_spine, dst)
                    else:
                        # The case that the current switch is the leaf switch connected to dst server
                        next_hop_list = original_next_hop_list
                        if dst in next_hop_list:
                            next_hop_list = [dst]
                    next_hop_list.sort()
                    next_hop = self._get_random_path(cur_node, next_hop_list, hop_list, hash_factor,
                                                     min(2, rehashing_time), src, dst, port_routing)
                    hop_list.append(next_hop)
                link_flow_occupy_num = len(infra.get_link_flow_occupy_list(cur_node, next_hop, taskid))
                link_capacity = infra.get_a_link_capacity(cur_node, next_hop, taskid)
                tmp_link_capacity = link_capacity / link_flow_occupy_num if link_flow_occupy_num > 0 else link_capacity
                if tmp_link_capacity < cur_capacity:
                    cur_capacity = tmp_link_capacity
                cur_node = next_hop
            if cur_capacity > best_capacity:
                best_capacity = cur_capacity
                best_hop_list = deepcopy(hop_list)
        return best_hop_list

#####
    def _get_up_down_routing_OCS_wECMP(self, src, dst, taskid, rehashing_time=5):
        """使用figret进行流量工程和不使用figret时使用的流量算法。"""
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        best_hop_list = []
        best_capacity = 0

        for hash_factor in range(rehashing_time):
            hop_list = []
            cur_node = src
            cur_capacity = 1000000
            src_pod = src // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
            dst_pod = dst // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))

            while cur_node != dst:
                if Simulator.is_GPU(cur_node):
                    cur_pod = cur_node // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
                elif Simulator.is_leaf_switch(cur_node):
                    cur_pod = (cur_node - infra.NIC_num) // (infra.leaf_switch_num // infra.pod_num)
                elif Simulator.is_spine_switch(cur_node):
                    cur_pod = (cur_node - infra.NIC_num - infra.leaf_switch_num) // (
                            infra.spine_switch_num // infra.pod_num)
                else:
                    raise ValueError("The node id is invalid: {}".format(cur_node))
                next_hop_list = []
                prob_list = []

                if cur_pod == src_pod:  # 当前在起点Pod中
                    if Simulator.is_GPU(cur_node):  # 当前在起点Pod的Nic中，选择leaf上行
                        to_dst_hop_list = Simulator.intra_pod_up_table[src]
                        next_hop = self._get_random_path(src, to_dst_hop_list, hop_list, hash_factor,
                                                         rehashing_time, self._src, self._dst)

                        cur_node = next_hop
                        hop_list.append(next_hop)
                        continue
                    elif Simulator.is_leaf_switch(cur_node):  # 当前在起点Pod的leaf中，按照比例选择spine上行或者到达目的地nic
                        if dst in Simulator.intra_pod_down_table[cur_node]:  # 下行转发来自同Pod其他leaf的流
                            # nic-leaf-nic case
                            hop_list.append(dst)
                            break
                        elif dst_pod == src_pod:
                            # nic-leaf-spine-leaf-nic case
                            next_hop_list = Simulator.intra_pod_up_table[cur_node]
                            hop_list.append(random.choice(next_hop_list))
                            src_pod = -1  # 为了避免进入第一个if分支
                            print("debug dragonfly",src,dst,cur_node,cur_pod, src_pod,hop_list)
                            continue
                        for spine_index in Simulator.intra_pod_up_table[cur_node]:
                            # 选择能够连接到目的Pod的spine
                            direct_table = Simulator.inter_pod_weighted_direct_table[spine_index]
                            if dst_pod in direct_table and len(direct_table[dst_pod]) != 0:
                                next_hop_list.append(spine_index)
                                next_hop_p = 0
                                for _, prob in direct_table[dst_pod]:
                                    next_hop_p += prob
                                prob_list.append(next_hop_p)
                            twohop_table = Simulator.inter_pod_weighted_twohop_table[spine_index]
                            if dst_pod in twohop_table and len(twohop_table[dst_pod]) != 0:
                                next_hop_p = 0
                                for _, prob in twohop_table[dst_pod]:
                                    next_hop_p += prob
                                if not next_hop_list or next_hop_list[-1] != spine_index:
                                    # 说明从当前spine到目的pod没有直连路径
                                    next_hop_list.append(spine_index)
                                    prob_list.append(next_hop_p)
                                else:
                                    # 概率等于直连路径+两跳路径的概率和
                                    prob_list[-1] += next_hop_p
                        # 验证权重是否为1
                        total_weight = sum(prob_list)
                        if not -1e-6 <= total_weight - 1 <= 1e-6:
                            # warnings.warn(
                            #     "The total weight is not equal to 1. "
                            #     "curr_node = {}, dst_pod = {}, prob_list = {}".format(cur_node, dst_pod, prob_list),
                            #     RuntimeWarning)
                            prob_list = [prob / total_weight for prob in prob_list]
                    else:  # 当前在起点Pod的spine中，按照比例选择
                        try:
                            if src == '1792' and dst == '1856':
                                print("debug dragonfly",cur_node,Simulator.inter_pod_weighted_direct_table[cur_node])
                            for next_hop, prob in Simulator.inter_pod_weighted_direct_table[cur_node][dst_pod]:
                                if src == '1792' and dst == '1856':
                                    print("debug dragonfly",next_hop)
                                next_hop_list.append(next_hop)
                                prob_list.append(prob)
                            for next_hop, prob in Simulator.inter_pod_weighted_twohop_table[cur_node][dst_pod]:
                                next_hop_list.append(next_hop)
                                prob_list.append(prob)
                            sum_p = sum(prob_list)
                            prob_list = [prob / sum_p for prob in prob_list]
                        except KeyError:
                            pass

                elif cur_pod != src_pod and cur_pod != dst_pod:  # 当前在中转Pod的spine中
                    try:
                        for next_hop, prob in Simulator.inter_pod_weighted_direct_table[cur_node][dst_pod]:
                            next_hop_list.append(next_hop)
                            prob_list.append(prob)
                    except KeyError:
                        pass
                    total_weight = sum(prob_list)
                    prob_list = [probability / total_weight for probability in prob_list]

                else:  # 当前在目的地Pod中
                    print("debug dragonfly",src,cur_node,dst,cur_pod, src_pod,hop_list)
                    for potential_node in Simulator.intra_pod_down_table[cur_node][dst]:
                        next_hop_list.append(potential_node)
                    next_hop = random.choice(next_hop_list)
                    hop_list.append(next_hop)
                    break
                if not next_hop_list:
                    raise ValueError("No available next hop!")
                if len(next_hop_list) == 1:
                    next_hop = next_hop_list[0]
                else:
                    # next_hop_list.sort()
                    next_hop = self._get_random_path_given_possibility(next_hop_list, prob_list,
                                                                       hop_list, hash_factor, src, dst)
                hop_list.append(next_hop)

                link_flow_occupy_num = len(infra.get_link_flow_occupy_list(cur_node, next_hop, taskid))
                link_capacity = infra.get_a_link_capacity(cur_node, next_hop, taskid)
                tmp_link_capacity = link_capacity / link_flow_occupy_num if link_flow_occupy_num > 0 else link_capacity
                if tmp_link_capacity < cur_capacity:
                    cur_capacity = tmp_link_capacity
                cur_node = next_hop

            if cur_capacity > best_capacity:
                # 优先选择带宽更大的路径
                best_capacity = cur_capacity
                best_hop_list = deepcopy(hop_list)
            elif -1e-6 <= cur_capacity - best_capacity <= 1e-6:
                # 相同带宽的情况下，选择路径更短的
                if len(hop_list) < len(best_hop_list):
                    best_hop_list = deepcopy(hop_list)
        return best_hop_list

    def _get_up_down_routing_OCS_wECMP_rail_optimized(self, src, dst, taskid, rehashing_time=5):
        """使用figret进行流量工程和不使用figret时使用的流量算法。"""
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        best_hop_list = []
        best_capacity = 0

        for hash_factor in range(rehashing_time):
            hop_list = []
            cur_node = src
            cur_capacity = 1000000
            src_pod = src // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
            dst_pod = dst // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))

            while cur_node != dst:
                if Simulator.is_GPU(cur_node):
                    cur_pod = cur_node // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
                elif Simulator.is_leaf_switch(cur_node):
                    cur_pod = (cur_node - infra.NIC_num) // (infra.leaf_switch_num // infra.pod_num)
                elif Simulator.is_spine_switch(cur_node):
                    cur_pod = (cur_node - infra.NIC_num - infra.leaf_switch_num) // (infra.spine_switch_num // infra.pod_num)
                else:
                    raise ValueError("The node id is invalid: {}".format(cur_node))
                next_hop_list = []
                prob_list = []
                
                if cur_pod == src_pod:  # 当前在起点Pod中
                    if Simulator.is_GPU(cur_node):  # 当前在起点Pod的GPU中，随机选择leaf上行
                        to_dst_hop_list = Simulator.intra_pod_up_table[src]
                        next_hop = self._get_random_path(src, to_dst_hop_list, hop_list, hash_factor,
                                                         rehashing_time, self._src, self._dst)
                        cur_node = next_hop
                        hop_list.append(next_hop)
                        continue
                    elif Simulator.is_leaf_switch(cur_node):  # 当前在起点Pod的leaf中，按照比例选择spine上行或者到达目的地nic
                        if dst in Simulator.intra_pod_down_table[cur_node]:  # 下行转发来自同Pod其他leaf的流
                            # nic-leaf-nic case
                            hop_list.append(dst)
                            break
                        for spine_index in Simulator.intra_pod_up_table[cur_node]:
                            # nic-leaf-spine-leaf-nic case
                            if dst in Simulator.intra_pod_down_table[spine_index]:
                                list_length = len(Simulator.intra_pod_down_table[spine_index][dst])
                                for potential_node in Simulator.intra_pod_down_table[spine_index][dst]:
                                    next_hop_list.append(potential_node)
                                    prob_list.append(1 / list_length)
                            # 选择连接到目的Pod的spine
                            elif dst_pod in Simulator.inter_pod_weighted_direct_table[spine_index] and \
                                    len(Simulator.inter_pod_weighted_direct_table[spine_index][dst_pod]) != 0:
                                next_hop_list.append(spine_index)
                                next_hop_p = 0
                                for _, probability in Simulator.inter_pod_weighted_direct_table[spine_index][dst_pod]:
                                    next_hop_p += probability
                                prob_list.append(next_hop_p)
                            elif dst_pod in Simulator.inter_pod_weighted_twohop_table[spine_index] and \
                                    len(Simulator.inter_pod_weighted_twohop_table[spine_index][dst_pod]) != 0:
                                next_hop_p = 0
                                for _, probability in Simulator.inter_pod_weighted_twohop_table[spine_index][dst_pod]:
                                    next_hop_p += probability
                                if not next_hop_list or next_hop_list[-1] != spine_index:
                                    # 说明从当前spine到目的pod没有直连路径
                                    next_hop_list.append(spine_index)
                                    prob_list.append(next_hop_p)
                                else:
                                    prob_list[-1] += next_hop_p
                        # 验证权重是否为1
                        total_weight = sum(prob_list)
                        prob_list = [probability / total_weight for probability in prob_list]
                        # assert -1e-6 <= total_weight - 1 <= 1e-6, "The total weight is not equal to 1."
                    else:  # 当前在起点Pod的spine中，按照比例选择
                        try:
                            for next_hop, probability in Simulator.inter_pod_weighted_direct_table[cur_node][dst_pod]:
                                next_hop_list.append(next_hop)
                                prob_list.append(probability)
                            for next_hop, probability in Simulator.inter_pod_weighted_twohop_table[cur_node][dst_pod]:
                                next_hop_list.append(next_hop)
                                prob_list.append(probability)
                            sum_p = sum(prob_list)
                            prob_list = [probability / sum_p for probability in prob_list]
                        except KeyError:
                            pass

                elif cur_pod != src_pod and cur_pod != dst_pod:  # 当前在中转Pod的spine中
                    try:
                        for next_hop, probability in Simulator.inter_pod_weighted_direct_table[cur_node][dst_pod]:
                            next_hop_list.append(next_hop)
                            prob_list.append(probability)
                    except KeyError:
                        pass
                    total_weight = sum(prob_list)
                    prob_list = [probability / total_weight for probability in prob_list]

                else:  # 当前在目的地Pod中
                    list_length = len(Simulator.intra_pod_down_table[cur_node][dst])
                    for potential_node in Simulator.intra_pod_down_table[cur_node][dst]:
                        next_hop_list.append(potential_node)
                        prob_list.append(1 / list_length)
                
                assert len(next_hop_list) > 0, "No available next hop!"

                if len(next_hop_list) == 1:
                    next_hop = next_hop_list[0]
                else:
                    # next_hop_list.sort()
                    next_hop = self._get_random_path_given_possibility(next_hop_list, prob_list, hop_list, hash_factor, src, dst)
                hop_list.append(next_hop)

                link_flow_occupy_num = len(infra.get_link_flow_occupy_list(cur_node, next_hop, taskid))
                link_capacity = infra.get_a_link_capacity(cur_node, next_hop, taskid)
                tmp_link_capacity = link_capacity / link_flow_occupy_num if link_flow_occupy_num > 0 else link_capacity
                if tmp_link_capacity < cur_capacity:
                    cur_capacity = tmp_link_capacity
                cur_node = next_hop

            if cur_capacity > best_capacity:
                # 优先选择带宽更大的路径
                best_capacity = cur_capacity
                best_hop_list = deepcopy(hop_list)
            elif -1e-6 <= cur_capacity - best_capacity <= 1e-6:
                # 相同带宽的情况下，选择路径更短的
                if len(hop_list) < len(best_hop_list):
                    best_hop_list = deepcopy(hop_list)
        return best_hop_list

    def _get_up_down_routing_OCS_ECMP(self, src, dst, taskid, rehashing_time=5):
        """
        现阶段使用ECMP不进行流量工程的Lumoscore的路由算法。
        """
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        best_hop_list = []
        best_capacity = 0
        port_routing = Simulator.CONF_DICT['port_routing'] == 'yes'
        # if 'is_two_iter' in Simulator.CONF_DICT and Simulator.CONF_DICT['is_two_iter'] == 'yes':
        #     rehashing_time = 10

        for hash_factor in range(rehashing_time):
            hop_list = []
            cur_node = src
            cur_capacity = 1000000
            src_pod = src // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
            dst_pod = dst // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
            while cur_node != dst:
                if len(hop_list) == 0:
                    # nic to leaf stage
                    next_hop_list = Simulator.intra_pod_up_table[src]
                elif len(hop_list) == 1:
                    # leaf to spine stage or leaf to nic stage
                    if dst in Simulator.intra_pod_down_table[cur_node]:
                        # nic-leaf-nic case
                        hop_list.append(dst)
                        break
                    else:
                        # select spine switches that can connect to the destination pod
                        next_hop_list = []
                        if src_pod == dst_pod:
                            for spine_index in Simulator.intra_pod_up_table[cur_node]:
                                next_hop_list.append(spine_index)
                        else:
                            for spine_index in Simulator.intra_pod_up_table[cur_node]:
                                if spine_index in Simulator.inter_pod_table and dst_pod in Simulator.inter_pod_table[spine_index]:
                                    next_hop_list.append(spine_index)
                elif len(hop_list) == 2:
                    # inter-pod routing stage or spine to leaf stage
                    if src_pod == dst_pod:
                        next_hop_list = Simulator.intra_pod_down_table[cur_node][dst].copy()
                    else:
                        next_hop_list = Simulator.inter_pod_table[cur_node][dst_pod].copy()
                else:
                    # down stage
                    next_hop_list = Simulator.intra_pod_down_table[cur_node][dst].copy()
                if len(next_hop_list) == 1:
                    next_hop = next_hop_list[0]
                else:
                    next_hop_list.sort()
                    next_hop = self._get_random_path(cur_node, next_hop_list, hop_list, hash_factor,
                                                     rehashing_time, src, dst, port_routing)
                hop_list.append(next_hop)

                link_flow_occupy_num = len(infra.get_link_flow_occupy_list(cur_node, next_hop, taskid))
                link_capacity = infra.get_a_link_capacity(cur_node, next_hop, taskid)
                tmp_link_capacity = link_capacity / link_flow_occupy_num if link_flow_occupy_num > 0 else link_capacity
                if tmp_link_capacity < cur_capacity:
                    cur_capacity = tmp_link_capacity
                cur_node = next_hop
            if cur_capacity > best_capacity:
                best_capacity = cur_capacity
                best_hop_list = deepcopy(hop_list)
        return best_hop_list

    def _get_up_down_routing_OCS_balance(self, src, dst):
        """
        LumosCore中的balance_routing算法。
        """
        from rapidAIsim.core.simulator import Simulator
        hop_list = []
        cur_node = src
        src_pod = src // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
        dst_pod = dst // (int(Simulator.CONF_DICT['NIC_num']) // int(Simulator.CONF_DICT['pod_num']))
        while cur_node != dst:
            if len(hop_list) == 0:
                # nic to leaf stage
                next_hop_list = Simulator.intra_pod_up_table[src]
            elif len(hop_list) == 1:
                # leaf to spine stage or leaf to nic stage
                if dst in Simulator.intra_pod_down_table[cur_node]:
                    # nic-leaf-nic case
                    hop_list.append(dst)
                    break
                else:
                    # select spine switches that can connect to the destination pod
                    next_hop_list = []
                    if src_pod == dst_pod:
                        for spine_index in Simulator.intra_pod_up_table[cur_node]:
                            next_hop_list.append(spine_index)
                    else:
                        for spine_index in Simulator.intra_pod_up_table[cur_node]:
                            if spine_index in Simulator.inter_pod_table and dst_pod in Simulator.inter_pod_table[spine_index]:
                                next_hop_list.append(spine_index)
            elif len(hop_list) == 2:
                # inter-pod routing stage or spine to leaf stage
                if src_pod == dst_pod:
                    next_hop_list = Simulator.intra_pod_down_table[cur_node][dst].copy()
                else:
                    next_hop_list = Simulator.inter_pod_table[cur_node][dst_pod].copy()
            else:
                # down stage
                next_hop_list = Simulator.intra_pod_down_table[cur_node][dst].copy()
            if len(next_hop_list) == 1:
                next_hop = next_hop_list[0]
            else:
                if Simulator.is_leaf_switch(cur_node):
                    next_hop_list = list(reversed(next_hop_list))
                else:
                    next_hop_list.sort()
                next_hop_list = list(set(next_hop_list))
                if len(next_hop_list) == 0:
                    print("debug next_hop_list",src, dst,cur_node,self._taskid,hop_list)
                next_hop = self._get_balance_path(cur_node, next_hop_list, hop_list,src,dst)
                # next_hop = self._get_static_routing_path(cur_node, next_hop_list, hop_list,src,dst)
            hop_list.append(next_hop)
            cur_node = next_hop
        return hop_list

    def _get_balance_path(self, tmp_src, to_dst_next_hop_list, hop_list,src=-1,dst=-1):
        """
        选择当前路径中剩余带宽最大的路径。
        :param tmp_src: 当前节点
        :param to_dst_next_hop_list: 下一跳节点列表
        :param hop_list: 已经选择的路径
        :return: 最佳下一跳节点
        """
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        residue_occupy_ratio_list = []
        for next_hop in to_dst_next_hop_list:
            if next_hop in hop_list:
                # Avoid repeat loop paths.
                continue
            the_link_flow_occupy_num = len(infra.get_link_flow_occupy_list(tmp_src, next_hop, self._taskid))

            the_link_capacity = infra.get_a_link_capacity(tmp_src, next_hop, self._taskid)

            if the_link_flow_occupy_num > 0:
                residue_occupy_ratio_list.append(the_link_capacity / the_link_flow_occupy_num)
            else:
                residue_occupy_ratio_list.append(the_link_capacity + 1)

        max_ratio = max(residue_occupy_ratio_list)
        size = len(residue_occupy_ratio_list)
        # print("debug balance at time",Simulator.get_current_time())
        # print(f'{src},{dst},{tmp_src}')
        # print(residue_occupy_ratio_list)
        # print(to_dst_next_hop_list)
        max_residue_occupy_ratio_index_list = [i for i in range(size) if residue_occupy_ratio_list[i] == max_ratio]
        max_residue_occupy_ratio_index = random.choice(max_residue_occupy_ratio_index_list)
        return to_dst_next_hop_list[max_residue_occupy_ratio_index]

    def _get_random_path(self, tmp_src, to_dst_next_hop_list, hop_list, need_routing_seed=0, rehash_max_time=0, src=0,
                         dst=0, port_routing=True):
        # Currently, if every NIC has multiple shortest paths to another NIC,
        # we select a shortest path randomly.
        best_next_hop = -1
        cur_contention_level = 100000
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        if port_routing:
            rdm = (need_routing_seed + src + dst) % len(to_dst_next_hop_list)
            return to_dst_next_hop_list[rdm]
        for iter_num in range(rehash_max_time):
            while_iter_num = 0
            while True:
                if 'flowletsize' not in Simulator.CONF_DICT or Simulator.CONF_DICT['flowletsize'] == 'MAX' or \
                        Simulator.CONF_DICT['flowletsize'] == '':
                    random.seed(self._flow_id)
                else:
                    random.seed(src + dst)
                hash_value = src + dst + need_routing_seed + iter_num
                rdm = mmh3.hash('foo', hash_value) % len(to_dst_next_hop_list)
                # rdm = hash_value % len(to_dst_next_hop_list)
                next_hop = to_dst_next_hop_list[rdm]
                if next_hop not in hop_list:
                    break
                while_iter_num += 1
                if while_iter_num > 10000:
                    raise RuntimeError('Cannot find a proper next hop after 10000 iteration.')
            link_num = int(infra.get_link_num(tmp_src, next_hop, -2))
            if link_num < cur_contention_level:
                cur_contention_level = link_num
                best_next_hop = next_hop
        return best_next_hop

    def _get_random_path_given_possibility(self, to_dst_next_hop_list, to_dst_next_hop_possibility_list, hop_list,
                                           hash_factor, src, dst):
        # Currently, if every NIC has multiple shortest paths to another NIC,
        # we select a shortest path randomly.
        from rapidAIsim.core.simulator import Simulator
        while_iter_num = 0
        while True:
            random.seed(src + dst + hash_factor)
            next_hop = random.choices(to_dst_next_hop_list, weights=to_dst_next_hop_possibility_list)[0]
            if next_hop not in hop_list:
                break
            while_iter_num += 1
            if while_iter_num > 10000:
                raise RuntimeError('Cannot find a proper next hop after 10000 iteration.')
        return next_hop

    def _get_static_routing_path(self, cur_src, to_dst_next_hop_list, hop_list, original_src,dst=-1):
        from rapidAIsim.core.simulator import Simulator
        cur_device = Simulator.get_infrastructure().get_device(cur_src)

        available_port_list = []
        for next_hop in to_dst_next_hop_list:
            if next_hop in hop_list:
                # Avoid repeat loop paths.
                continue
            available_port_list.append(next_hop)

        available_port_num = len(available_port_list)
        static_next_hop = available_port_list[original_src % available_port_num]
        # print("debug static routing at time",Simulator.get_current_time())
        # print(f'{original_src},{dst},{cur_src},{static_next_hop},')
        # print(original_src % available_port_num)
        # print(to_dst_next_hop_list)
        return static_next_hop

    # def _get_static_routing_path(self, cur_src, to_dst_next_hop_list, hop_list, original_src,dst=-1):
    #     from rapidAIsim.core.simulator import Simulator
    #     cur_device = Simulator.get_infrastructure().get_device(cur_src)

    #     available_port_list = []
    #     for next_hop in to_dst_next_hop_list:
    #         if next_hop in hop_list:
    #             # Avoid repeat loop paths.
    #             continue
    #         available_port_list += cur_device.get_port_list(next_hop)

    #     available_port_num = len(available_port_list)
    #     static_port = available_port_list[original_src % available_port_num]
    #     static_next_hop = cur_device.get_target_device_id(static_port)
    #     print("debug static routing at time",Simulator.get_current_time())
    #     print(f'{original_src},{dst},{cur_src},{static_next_hop}')
    #     print(original_src % available_port_num)
    #     print(to_dst_next_hop_list)
    #     return static_next_hop

    def get_interAS_hop_list(self):
        """
        在已有hop_list的情况下，获取跨AS的路径（跨AS指的是二层拓扑中跨leaf，三层拓扑中跨Pod）。
        该函数仅用于统计流量矩阵，不用于实际传输的路由寻路。
        """
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        src = self._src
        interAS_hop_list = []
        if not self._hop_list or self._hop_list[0] == src:
            return interAS_hop_list
        if infra.layers == 2:
            nic_num_per_leaf = infra.leaf_switch_port_num // 2
            src_leaf = src // nic_num_per_leaf
            for dst in self._hop_list:
                if Simulator.is_GPU(dst):
                    dst_leaf = dst // nic_num_per_leaf
                elif Simulator.is_leaf_switch(dst):
                    dst_leaf = dst - infra.NIC_num
                else:  # case spine switch is skipped
                    continue
                if src_leaf != dst_leaf and (src_leaf, dst_leaf, 0) not in interAS_hop_list:
                    interAS_hop_list.append((src_leaf, dst_leaf, 0))
                src_leaf = dst_leaf
        elif infra.layers == 3:
            # hop_list = self._hop_list
            # if len(hop_list) != 5:
            #     return []
            # for dst in self._hop_list:
            #     if Simulator.is_spine_switch(dst):
            #         dst_spine = dst - infra.NIC_num - infra.leaf_switch_num
            #         interAS_hop_list.append(dst_spine)
            #     else:  # core switch case is skipped
            #         continue
            # assert len(interAS_hop_list) == 2, "InterAS hop list length error!"
            # interAS_hop_list = [(interAS_hop_list[0], interAS_hop_list[1])]
            spine_num_per_pod = infra.spine_switch_num // infra.pod_num
            src_pod = -1
            for dst in self._hop_list:
                if not Simulator.is_spine_switch(dst):
                    continue
                dst_spine = dst - infra.NIC_num - infra.leaf_switch_num
                dst_pod = dst_spine // spine_num_per_pod
                spine_index_in_pod = dst_spine % spine_num_per_pod
                spine_index_in_pod = 0 #TODO之后修改一下，目前是Pod level，所以可以0
                if src_pod != -1 and (src_pod, dst_pod, spine_index_in_pod) not in interAS_hop_list:
                    # print(src_pod, dst_pod, spine_index_in_pod)
                    interAS_hop_list.append((src_pod, dst_pod, spine_index_in_pod))
                src_pod = dst_pod
        return interAS_hop_list

    def get_hop_list(self):
        """Get the path from src to dst.
        All the elements are feasible next hop.
        The last element in hop_list is dst.
        """
        return self._hop_list

    def set_hop_list(self, hop_list):
        from rapidAIsim.core.simulator import Simulator
        if self._conservative is True:
            tmp_src = self._src
            for next_hop in hop_list:
                Simulator.add_link_occupied_for_tasks(self._taskid, tmp_src, next_hop)
                tmp_src = next_hop

        self._hop_list = hop_list

    def get_flow_id(self):
        return self._flow_id

    def get_last_calculated_time(self):
        return self._last_calculated_time

    def set_last_calculated_time(self, sim_time):
        self._last_calculated_time = sim_time

    def set_remainder_size(self, remainder_size):
        self._remainder_size = remainder_size

    def get_remainder_size(self):
        return self._remainder_size

    def set_start_time(self, start_time):
        self._start_time = start_time

    def get_start_time(self):
        return self._start_time

    def get_src(self):
        return self._src

    def get_dst(self):
        return self._dst

    def get_size(self):
        return self._size

    def get_total_round(self):
        return self._total_round

    def get_task_id(self):
        return self._taskid

    def get_round_id(self):
        return self._round_id

    def get_min_available_capacity(self):
        return self._min_available_capacity

    def set_min_available_capacity(self, min_available_capacity):
        self._min_available_capacity = min_available_capacity

    def get_expected_finish_time(self):
        return self._expected_finish_time

    def set_expected_finish_time(self, expected_finish_time):
        self._expected_finish_time = expected_finish_time

    def set_in_the_same_server(self):
        self._in_the_same_server_flag = True

    def is_in_the_same_server(self):
        return self._in_the_same_server_flag

    def set_need_recalculation(self, flag):
        self._need_recalculation = flag

    def whether_need_recalculation(self):
        return self._need_recalculation

    @property
    def last_statistic_time(self):
        return self._last_statistic_time

    @last_statistic_time.setter
    def last_statistic_time(self, t):
        self._last_statistic_time = t

    @property
    def traffic_matrix_remainder_size(self):
        return self._traffic_matrix_remainder_size

    @traffic_matrix_remainder_size.setter
    def traffic_matrix_remainder_size(self, size):
        self._traffic_matrix_remainder_size = size

    @property
    def need_prior_computation(self):
        return self._need_prior_calculate

    @need_prior_computation.setter
    def need_prior_computation(self, flag):
        self._need_prior_calculate = flag

    @property
    def need_subsequent_computation(self):
        return self._need_subsequent_calculate

    @need_subsequent_computation.setter
    def need_subsequent_computation(self, flag):
        self._need_subsequent_calculate = flag
