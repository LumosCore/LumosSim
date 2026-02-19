import math
import numpy as np
import logging
from rapidAIsim.scheduler.ocsexpander.ocsexpander_superpod import OCSExpander_superPod
import rapidAIsim.scheduler.elescheduler.gpu_placement as gpu_placement
from rapidAIsim.core.event.RepairEvent import RepairEvent
from rapidAIsim.core.network_refresh import handle_task_finish_immediately

logging.basicConfig(level=logging.ERROR)


class ELEExpander_SuperPod:
    def __init__(self):
        # static variable
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        self.pod_num = infra.pod_num
        self.spine_num_per_pod = infra.spine_switch_num // infra.pod_num
        self.spine_up_port_num = infra.spine_switch_port_num
        self.gpu_per_server = infra.NIC_num_in_a_server
        self.server_num_per_pod = infra.server_num // infra.pod_num
        self.server_per_leaf = infra.server_per_leaf
        self.TP = self.gpu_per_server
        self.leaf_per_pod = self.server_num_per_pod // infra.server_per_leaf
        # dynamical variable
        self.T_a_b = np.zeros((self.pod_num, self.pod_num), dtype=int)
        # self.cur_real_link_demand = np.zeros((self.pod_num, self.pod_num), dtype=int)
        self.job_flow_demand_map = {}
        self.gpu_per_pod = self.gpu_per_server * self.server_num_per_pod

        self.gpu_placement_scheduler = gpu_placement.GPUPlacement()
        self.allocated_link_mapping = self.translate_link_new()
        Simulator.reconfigure(self.allocated_link_mapping, 0)

        self.checkpoint_time = 60
        self.job_start_time_map = {}
        self.failure_id_gpu_status_map = {}

    def schedule(self, TP, DP, PP, EP, task_id,model_size):
        from rapidAIsim.core.simulator import Simulator
        Simulator.task_has_computation_time[task_id] = 0.
        Simulator.task_has_communication_size[task_id] = 0.
        Simulator.task_need_comp_time[task_id] = 0.
        Simulator.task_need_comm_size[task_id] = 0.
        Simulator.task_expected_comm_time[task_id] = 0.
        Simulator.task_actual_comm_time[task_id] = 0.
        print(f'task {task_id} with gpu size {TP * DP * PP} and EP is {EP}')
        # Step 1 gpu placement
        require_server_num = TP * DP * PP // self.gpu_per_server
        if EP > self.gpu_per_pod:
            pod_used_server_list, job_gpu_used_list, cur_ep_size = self.gpu_placement_scheduler.occupy_resource_for_large_EP(
            require_server_num, EP, task_id)
        else:
            pod_used_server_list, job_gpu_used_list = self.gpu_placement_scheduler.occupy_resource(require_server_num, EP,
                                                                                               task_id)
        if len(pod_used_server_list) == 0:
            return False, None, None, None, None

        # Step 2 flow demand generator
        if EP > self.gpu_per_pod:
            ep_dp_pp_array = OCSExpander_superPod.generate_flow_demand(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list)
            ep_qp, pp_qp, dp_qp, ep_link_demand = OCSExpander_superPod.generate_qp(ep_dp_pp_array, TP, DP, PP, EP, cur_ep_size, self.gpu_per_pod)
        else:
            dp_pp_array, ep_array = OCSExpander_superPod.generate_flow_demand_smallEP(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list)
            ep_qp, pp_qp, dp_qp = OCSExpander_superPod.generate_qp_smallEP(ep_array, dp_pp_array, TP, DP, PP, EP)
        # Step 3 generate flow
        allocated_link_mapping = self.allocated_link_mapping
        self.job_start_time_map[task_id] = Simulator.get_current_time()
        return True, ep_qp, pp_qp, dp_qp, allocated_link_mapping

    def random_fail_gpu(self, failure_id):
        # failure_id = -1*failure_id # gpus occupied by task i -1 is in failure
        new_banned_gpu_status, influenced_task_id, new_banned_server_per_pod = \
            self.gpu_placement_scheduler.random_fail_gpu(failure_id)
        if influenced_task_id != -1:
            # change remain model size of influenced_task
            from rapidAIsim.core.simulator import Simulator
            from rapidAIsim.task.waiting_task import WaitingTask
            Simulator.need_immediately_finish_task.append(influenced_task_id)
            handle_task_finish_immediately(influenced_task_id)
            # 任务相关流立即完成，同时任务再次进入等待队列
            task = Simulator.TASK_LIST[influenced_task_id]
            has_comp_ratio = Simulator.task_has_computation_time[influenced_task_id] / \
                             Simulator.task_need_comp_time[influenced_task_id]
            has_comm_ratio = Simulator.task_has_communication_size[influenced_task_id] / \
                             Simulator.task_need_comm_size[influenced_task_id]
            print(f'task {influenced_task_id} need comp {Simulator.task_need_comp_time[influenced_task_id]} '
                  f'need comm {Simulator.task_need_comm_size[influenced_task_id]}')
            print(f'task {influenced_task_id} has ratio {has_comp_ratio},{has_comm_ratio} '
                  f'has comp {Simulator.task_has_computation_time[influenced_task_id]} has comm '
                  f'{Simulator.task_has_communication_size[influenced_task_id]}')
            if has_comp_ratio > 1:
                print("debug has_comp_ratio", has_comp_ratio)
            assert has_comp_ratio <= 1.1
            if has_comm_ratio > 1:
                print("debug has_comp_ratio", has_comm_ratio)
            assert has_comm_ratio <= 1.1
            has_comp_ratio = min(has_comp_ratio, 0.9999)
            has_comm_ratio = min(has_comm_ratio, 0.9999)

            comm_time = task.duration_time - task.computation_time * task.computation_round
            remain_duration_time = (task.duration_time - has_comp_ratio * task.computation_time * task.computation_round
                                    - has_comm_ratio * comm_time)
            task.duration_time = remain_duration_time

            a_waiting_task = WaitingTask(Simulator.get_current_time(), task.model_size, task.gpu_num,
                                         Simulator.task_type_map[influenced_task_id], influenced_task_id, 0)
            Simulator.push_a_waiting_task(a_waiting_task)

        return new_banned_gpu_status, new_banned_server_per_pod

    def handle_failure_event(self, failure_id, duration_time):
        new_banned_gpu_status, new_banned_server_per_pod = self.random_fail_gpu(failure_id)
        if len(new_banned_gpu_status) > 0:
            self.failure_id_gpu_status_map[failure_id] = (new_banned_gpu_status, new_banned_server_per_pod)
            # 生成任务恢复事件
            from rapidAIsim.core.simulator import Simulator
            Simulator.register_event(
                RepairEvent(
                    duration_time,
                    failure_id,
                )
            )

    def handle_repair_event(self, failure_id):
        self.gpu_placement_scheduler.repair_fail_gpu(self.failure_id_gpu_status_map[failure_id][0],
                                                     self.failure_id_gpu_status_map[failure_id][1])

    @staticmethod
    def translate_link():
        # output
        from rapidAIsim.core.simulator import Simulator
        allocated_link_mapping = []

        total_port_num = int(Simulator.CONF_DICT['NIC_num'])

        max_spine_size_each_layer = int(Simulator.CONF_DICT['spine_switch_num'])
        port_per_spine = int(Simulator.CONF_DICT['spine_switch_port_num']) // 2
        tmp_pod_num = int(Simulator.CONF_DICT['NIC_num'])
        tmp_pod_port_size = total_port_num // tmp_pod_num
        stage_num = 0
        layer_num = math.ceil(math.log(total_port_num, port_per_spine))
        over_subscription = 1
        if 'over_subscription' in Simulator.CONF_DICT and len(Simulator.CONF_DICT['over_subscription']) > 0:
            over_subscription = max(1, int(Simulator.CONF_DICT['over_subscription']))

        while tmp_pod_num > 1:
            tmp_pod_port_size *= port_per_spine
            tmp_pod_num = max(1, total_port_num // tmp_pod_port_size)
            base_node_id = max(0, (stage_num - 1)) * max_spine_size_each_layer + min(1, stage_num) * total_port_num
            next_base_node_id = max(0, stage_num) * max_spine_size_each_layer + total_port_num
            if stage_num == 0:
                cur_layer_size = total_port_num
            # elif stage_num == layer_num-1:
            #     cur_layer_size = max_spine_size_each_layer // over_subscription
            else:
                cur_layer_size = max_spine_size_each_layer
            for tmp_node_id in range(cur_layer_size):
                node_per_pod = int(cur_layer_size) // tmp_pod_num
                spine_per_pod = int(max_spine_size_each_layer) // tmp_pod_num
                if stage_num == 0:
                    port_per_node = 1
                elif stage_num == layer_num - 1:
                    port_per_node = port_per_spine // over_subscription
                else:
                    port_per_node = port_per_spine
                if stage_num == 0:
                    tmp_start_node_id = tmp_node_id // node_per_pod
                elif stage_num == layer_num - 1:
                    tmp_start_node_id = tmp_node_id // spine_per_pod * spine_per_pod + ((port_per_spine * (
                            tmp_node_id % spine_per_pod)) % spine_per_pod) // 2  # 第几个pod的第一个node,pod内的第几个node
                else:
                    tmp_start_node_id = tmp_node_id // spine_per_pod * spine_per_pod + ((port_per_spine * (
                            tmp_node_id % spine_per_pod)) % spine_per_pod)  # 第几个pod的第一个node,pod内的第几个node
                for tmp_node_port in range(int(port_per_node)):
                    allocated_link_mapping.append(
                        [base_node_id + tmp_node_id, next_base_node_id + tmp_start_node_id + tmp_node_port, 1])
                    allocated_link_mapping.append(
                        [next_base_node_id + tmp_start_node_id + tmp_node_port, base_node_id + tmp_node_id, 1])

            stage_num += 1
        return allocated_link_mapping

    @staticmethod
    def translate_link_new():
        from rapidAIsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        rail_optimized = Simulator.CONF_DICT['rail_optimized'] == 'yes'

        total_gpu_num = infra.NIC_num_in_a_server * infra.server_num
        switch_port_num = infra.leaf_switch_port_num
        gpu_num_per_server = infra.NIC_num_in_a_server
        leaf_switch_num = infra.leaf_switch_num
        # generate gpu->leaf table
        connection_info_list = ELEExpander_SuperPod.generate_gpu_leaf_table(
            total_gpu_num, switch_port_num, gpu_num_per_server, rail_optimized)

        prev_layer_pod_up_port_num = switch_port_num // 2
        # generate other layers' tables
        try:
            over_subscription = max(1, int(Simulator.CONF_DICT['over_subscription']))
        except (KeyError, ValueError):
            over_subscription = 1

        # 下一层是否是最顶层
        is_top_layer = False
        i = 0
        while not is_top_layer:
            pod_up_port_num = prev_layer_pod_up_port_num * switch_port_num
            if pod_up_port_num >= total_gpu_num:
                is_top_layer = True
                pod_num = 1
            else:
                pod_up_port_num //= 2
                pod_num = total_gpu_num // pod_up_port_num
            next_layer_start_node_id = total_gpu_num + leaf_switch_num * (i + 1)
            curr_layer_start_node_id = total_gpu_num + leaf_switch_num * i
            if is_top_layer:
                next_layer_node_down_port_num = switch_port_num
            else:
                next_layer_node_down_port_num = switch_port_num // 2

            info_list = ELEExpander_SuperPod.generate_aggregation_table(
                pod_num, switch_port_num // 2, leaf_switch_num, next_layer_start_node_id,
                curr_layer_start_node_id, next_layer_node_down_port_num, over_subscription)
            connection_info_list.extend(info_list)
            prev_layer_pod_up_port_num = pod_up_port_num
            i += 1
        return connection_info_list

    @staticmethod
    def generate_gpu_leaf_table(total_gpu_num, leaf_port_num, gpu_per_server, rail_optimized=False):
        """
        Generate the gpu->leaf table.
        :param total_gpu_num: the total number of GPUs in the cluster
        :param leaf_port_num: the number of ports in a leaf switch (including up and down ports)
        :param gpu_per_server: the number of GPUs in a server
        :param rail_optimized: whether the leaf switch is rail optimized
        """
        from rapidAIsim.core.simulator import Simulator
        leaf_up_port_num = leaf_port_num // 2
        leaf_start_id = total_gpu_num
        connection_info_list = []
        for i in range(total_gpu_num):
            nic_server_index = i // gpu_per_server
            if rail_optimized:
                nic_rail_index = nic_server_index // leaf_up_port_num  # nic所在rail编号
                leaf_index = nic_rail_index * gpu_per_server + i % gpu_per_server
                leaf_index += leaf_start_id
            else:
                leaf_index = leaf_start_id + i // leaf_up_port_num
            Simulator.clos_up_table[i].append(leaf_index)
            Simulator.clos_down_table[leaf_index][i].append(i)
            connection_info_list.append([i, leaf_index, 1])
            connection_info_list.append([leaf_index, i, 1])
        return connection_info_list

    @staticmethod
    def generate_aggregation_table(pod_num, node_up_port_num, curr_layer_node_num, next_layer_start_node_id,
                                   curr_layer_start_node_id, next_layer_node_down_port_num, over_subscription):
        """
        Generate the aggregation table.
        :param pod_num: the number of pods
        :param node_up_port_num: the number of up ports in a node
        :param curr_layer_node_num: the number of nodes in the current layer
        :param next_layer_start_node_id: the start id of the next layer
        :param curr_layer_start_node_id: the start id of the current layer
        :param next_layer_node_down_port_num: the number of down ports in a node in the next layer
        :param over_subscription: the over-subscription ratio
        """
        from rapidAIsim.core.simulator import Simulator
        connection_info_list = []
        pod_node_num = curr_layer_node_num // pod_num
        if pod_num == 1:
            pod_up_port_num = node_up_port_num * pod_node_num // over_subscription
            next_layer_node_num = pod_up_port_num * over_subscription * pod_num // next_layer_node_down_port_num
        else:
            pod_up_port_num = node_up_port_num * pod_node_num
            next_layer_node_num = pod_up_port_num * pod_num // next_layer_node_down_port_num
        group_node_num = next_layer_node_num // pod_num  # 下一层每个group的节点数

        for i in range(pod_num):
            for j in range(pod_up_port_num):
                node_index = curr_layer_start_node_id + i * pod_node_num + j // node_up_port_num
                next_hop_index = next_layer_start_node_id + i * group_node_num + j % group_node_num
                Simulator.clos_up_table[node_index].append(next_hop_index)
                for dst in Simulator.clos_down_table[node_index].keys():
                    Simulator.clos_down_table[next_hop_index][dst].append(node_index)
                connection_info_list.append([node_index, next_hop_index, 1])
                connection_info_list.append([next_hop_index, node_index, 1])
        return connection_info_list

    def update_finished_job(self, taskid, sim_time, waiting_list):
        # Step 1 update demand
        # self.T_a_b -= self.job_flow_demand_map[taskid]
        # Step 2 release gpu
        self.gpu_placement_scheduler.release_resource(taskid)
