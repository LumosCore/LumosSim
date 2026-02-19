import argparse
import random
import heapq
import math
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from rapidAIsim.core import *
from rapidAIsim.conf.global_conf import GlobalConf
from rapidAIsim.core.event.load_failure_info_and_generate_failure_event import generate_failure_event
from rapidAIsim.task.task_info import TaskInfo
from collections import defaultdict

class Simulator:
    """
    
    Simulator takes charge of the global management of
    config, simulation time, events and random module etc.
    So its member functions are mostly static methods.
    """

    FLOWID = 0

    _task_record_dict: Dict[int, List[int]] = {}  # {'taskid': flowid}

    _wait_transmit_dict: Dict[str, Dict[int, Flow]] = {}  # {'taskid_roundid': {'flowid', Flow}, ...}

    _finish_transmit_dict = {}

    WAITING_TASK_LIST = []  # [WaitingTask obj, WaitingTask obj, ...] should be used through heapq.

    TASK_LIST: List[TaskInfo] = []

    ITERATION_FINISH_ROUNDID_DICT = {}  # {'taskid': [finish-flag roundid, finish-flag roundid, ...]}

    FLOW_COMPLETION_EVENT_RECORD = None

    LINK_OCCUPIED_FOR_TASKS = {}

    BEFOREHAND_PATH = {}  # {(src,dst): [hops], ...}

    LAST_MIN_FINISH_TIME = None

    NEED_MODIFIED_FLOWIDS = set()

    TASK_SWITCH_DICT = {}  # {taskid: switch_list on the top of mesh_scheduler} for mesh_scheduler

    TASK_NIC_DICT = {}  # {taskid: need_NIC_list on the top of mesh_scheduler} for mesh_scheduler

    _global_conf = None

    CONF_DICT = {}

    SCHEDULER_TIME_COST = {}

    _infra_base = None

    _scheduler = None

    GPU_status = {}

    flow_whether_can_rerouting = {}

    flow_has_rehashing_time = {}

    contention_link = {}

    flow_latest_rerouting_time = {}

    flow_has_started = {}

    task_class = {}

    clos_up_table = defaultdict(list)

    clos_down_table = defaultdict(lambda: defaultdict(list))

    intra_pod_up_table = {}

    intra_pod_down_table = {}

    inter_pod_table = {}

    inter_pod_weighted_direct_table: Dict[int, Dict[int, List[Tuple[int, float]]]] = {}

    inter_pod_weighted_twohop_table: Dict[int, Dict[int, List[Tuple[int, float]]]] = {}

    subsequent_time = 0

    prior_time = 0

    _current_time = 0

    _event_q: List[Event] = []

    task_time_logger = None

    task_comm_ratio_logger = None

    # occupied_num_logger = None

    # task_queue_length_logger = None

    traffic_statistic_logger = None

    traffic_matrix_pod_level = None  # 用于统计3层拓扑下的pod间流量

    traffic_matrix_spine_level = None  # 用于统计3层拓扑下的spine间流量

    traffic_matrix_leaf_level = None  # 用于统计2层拓扑下的leaf间流量

    figret = None

    figret_toe = None

    history_traffic_matrix = []

    split_ratios = None

    task_type_map = {}

    need_immediately_finish_task = []

    task_has_computation_time = defaultdict(float)

    task_has_communication_size = defaultdict(float)

    task_need_comm_size = defaultdict(float)

    task_need_comp_time = defaultdict(float)

    task_expected_comm_time = defaultdict(float)

    task_actual_comm_time = defaultdict(float)
    
    rerouting_link = defaultdict(int)

    def __init__(self) -> None:
        raise Exception("Simulator acts as global static class and should not be instantiated!")

    @staticmethod
    def setup(conf_handler):
        """The setting up of simulation stage.
        Set: 
            - the global random seed.
            - simulation time.
            - event queue.
        """
        if conf_handler['Parameter'].get('seed') is None:
            Simulator._set_random_seed(999)
        else:
            Simulator._set_random_seed(conf_handler['Parameter'].get('seed'))

        Simulator._global_conf = GlobalConf(conf_handler)

        conf_parameter = Simulator._global_conf.config_handler['Parameter']
        for item in conf_parameter:
            Simulator.CONF_DICT[item] = conf_parameter[item]
        conf_task = Simulator._global_conf.config_handler['Task']
        for item in conf_task:
            Simulator.CONF_DICT[item] = conf_task[item]
        conf_topology = Simulator._global_conf.config_handler['Topology']
        for item in conf_topology:
            Simulator.CONF_DICT[item] = conf_topology[item]

        if Simulator.CONF_DICT.get('sync_delay_alpha') is None or Simulator.CONF_DICT.get('sync_delay_alpha') == '':
            Simulator.CONF_DICT['sync_delay_alpha'] = 0

        if Simulator.CONF_DICT['reconfiguration'] not in ['yes', 'no']:
            raise ValueError('Need to explicitly set reconfiguration')

        if Simulator.CONF_DICT['failure_interval'] != '':
            failure_interval = int(Simulator.CONF_DICT.get('failure_interval'))
            generate_failure_event(10000, failure_interval, 1000, 10000, 0.9)

    @staticmethod
    def create_infrastructure():

        Simulator._infra_base = InfraBase()

        connect_info_list = Simulator.get_global_conf().get_connect_info_list()

        # Create network topology
        Simulator._infra_base.create_topology(connect_info_list)
        print('Create topology.', flush=True)

        scheduler_type = Simulator.CONF_DICT['joint_scheduler']
        if scheduler_type in ['static', 'hw_eps_all2all', 'hw_eps_all2all2',
                              'hw_eps_all2all_old', 'hw_eps_all2all_hierarchical', 'static_scheduler_small']:
            # Find all paths in advance
            print('Finding all paths in advance start.', flush=True)
            # -2 means no reconfiguration
            Simulator._infra_base.find_all_path(-2)
            print('Finding all paths in advance is done.', flush=True)

        allow_type_list = [
            'NaiveScheduler', 'NaiveSchedulerCompare',
            'static', 'static_scheduler', 'static_scheduler_small',
            'GPUPlacementer', 'GPUPlacementer2', 'GPUPlacementer3', 'GPUPlacementer4', 'StaticPlacementer',
            'StaticPlacementerAI', 'StaticPlacementerRelax',
            'OCSExpander', 'OCSExpander_superPod','ELEExpander','ELEExpander_SuperPod',
            'hw_eps_all2all', 'hw_eps_all2all_old', 'hw_eps_all2all2',
            'hw_oxc_all2all', 'hw_oxc_all2all2', 'hw_oxc_all2all_sz',
            'hw_eps_all2all_hierarchical', 'hw_oxc_allreduce', 'hw_oxc_hdallreduce',
        ]
        # print("debug scheduler_type", scheduler_type)
        if scheduler_type not in allow_type_list:
            raise ValueError(f"Scheduler type {scheduler_type} is not supported.")
        # assert scheduler_type in allow_type_list

        # if scheduler_type in ['ELEExpander']:
        #     Simulator.create_clos_map()

    @staticmethod
    def create_clos_map():
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
                    if base_node_id + tmp_node_id not in Simulator.clos_up_table:
                        Simulator.clos_up_table[base_node_id + tmp_node_id] = []
                    Simulator.clos_up_table[base_node_id + tmp_node_id].append(
                        next_base_node_id + tmp_start_node_id + tmp_node_port)
                    if stage_num == 0:
                        Simulator.clos_down_table[tmp_node_id] = {}
                        Simulator.clos_down_table[tmp_node_id][tmp_node_id] = [tmp_node_id]
                        if next_base_node_id + tmp_start_node_id + tmp_node_port not in Simulator.clos_down_table:
                            Simulator.clos_down_table[
                                next_base_node_id + tmp_start_node_id + tmp_node_port] = {}
                        if base_node_id + tmp_node_id not in Simulator.clos_down_table[
                                next_base_node_id + tmp_start_node_id + tmp_node_port]:
                            Simulator.clos_down_table[
                                next_base_node_id + tmp_start_node_id + tmp_node_port][base_node_id + tmp_node_id] = [
                                base_node_id + tmp_node_id]
                    else:
                        if next_base_node_id + tmp_start_node_id + tmp_node_port not in Simulator.clos_down_table:
                            Simulator.clos_down_table[
                                next_base_node_id + tmp_start_node_id + tmp_node_port] = {}
                        for potential_dst in range(total_port_num):
                            if potential_dst in Simulator.clos_down_table[base_node_id + tmp_node_id]:
                                if potential_dst not in Simulator.clos_down_table[
                                        next_base_node_id + tmp_start_node_id + tmp_node_port]:
                                    Simulator.clos_down_table[
                                        next_base_node_id + tmp_start_node_id + tmp_node_port][potential_dst] = []
                                Simulator.clos_down_table[
                                    next_base_node_id + tmp_start_node_id + tmp_node_port][potential_dst].append(
                                    base_node_id + tmp_node_id)

            stage_num += 1

    @staticmethod
    def reset_traffic_matrix():
        # 三层拓扑才会记录下面的流量矩阵，多出的一个维度是为了输出时格式能够统一
        pod_num = Simulator._infra_base.pod_num
        spine_num = Simulator._infra_base.spine_switch_num
        Simulator.traffic_matrix_pod_level = np.zeros((pod_num, pod_num, 1), dtype=np.float64)
        Simulator.traffic_matrix_spine_level = np.zeros((pod_num, pod_num, spine_num // pod_num), dtype=np.float64)

        # 二层拓扑才会记录下面的流量矩阵
        leaf_num = Simulator._infra_base.leaf_switch_num
        Simulator.traffic_matrix_leaf_level = np.zeros((leaf_num, leaf_num, 1), dtype=np.float64)

    @staticmethod
    def get_traffic_matrix():
        config = Simulator.CONF_DICT
        if config['layers'] == '2':
            return Simulator.traffic_matrix_leaf_level
        elif config['layers'] == '3':
            if config['traffic_matrix_level'] == 'spine':
                return Simulator.traffic_matrix_spine_level
            else:  # 默认设置统计为spine级
                return Simulator.traffic_matrix_pod_level
        else:
            raise ValueError('Unsupported layers')

    @staticmethod
    def reconfigure(connect_info_list, taskid):
        Simulator._infra_base.reconfigure_topo(connect_info_list, taskid)
        Simulator._infra_base.find_all_path(taskid)
        print('Update topology and path dict are done!', flush=True)

    @staticmethod
    def get_infrastructure() -> InfraBase:
        return Simulator._infra_base

    @staticmethod
    def reset():
        Simulator._current_time = 0
        Simulator._event_q.clear()

    @staticmethod
    def core_run():
        while len(Simulator._event_q) > 0:
            event = heapq.heappop(Simulator._event_q)
            if not event.is_active:
                continue
            Simulator._current_time = event.event_time
            event.do_sth()
            refresh_completion_event()
            Simulator.flush_logger()
        if len(Simulator.WAITING_TASK_LIST) > 0:
            from rapidAIsim.core.stage_controller import _detect_and_trigger_a_task
            _detect_and_trigger_a_task()
            Simulator.core_run()

    @staticmethod
    def time_tick_based_core_run(time_tick):
        while len(Simulator._event_q) > 0:
            while len(Simulator._event_q) > 0:
                # Events in the same time do not boost time.
                event = heapq.heappop(Simulator._event_q)
                event_time = event.event_time
                if event_time <= Simulator._current_time:
                    event.do_sth()
                else:
                    Simulator.register_event(event)
                    break
            detect_and_update_flow_every_tick(time_tick)

            if event_time > Simulator._current_time:
                Simulator._current_time += time_tick

        while (len(Simulator.WAITING_TASK_LIST) > 0 or
               len(Simulator._infra_base.get_flow_infly_info_dict()) > 0
               or not Simulator.value_in_wait_transmit_dict_is_empty()):
            from rapidAIsim.core.stage_controller import _detect_and_trigger_a_task
            detect_and_update_flow_every_tick(time_tick)
            _detect_and_trigger_a_task()
            Simulator._current_time += time_tick

    @staticmethod
    def register_event(event: Event):
        heapq.heappush(Simulator._event_q, event)

    @staticmethod
    def get_current_time():
        """Return current simulation time.
        """
        return Simulator._current_time

    @staticmethod
    def get_plan_event_time(relative_time_from_now):
        """Return the time of triggering the event,
        that is Simulator.get_current_time() + relative_time_from_now.
        This is used to plan events in the future.
        """
        return Simulator._current_time + relative_time_from_now

    @staticmethod
    def _set_random_seed(seed):
        random.seed(seed)

    @staticmethod
    def get_global_conf():
        return Simulator._global_conf

    @staticmethod
    def get_task_record_dict():
        """
        {'taskid': flowid, ...}
        """
        return Simulator._task_record_dict

    @staticmethod
    def add_flowid_into_task_record(flowid, task_id):
        if Simulator._task_record_dict.get(task_id):
            Simulator._task_record_dict[task_id].append(flowid)
        else:
            Simulator._task_record_dict[task_id] = [flowid]

    @staticmethod
    def del_flowid_from_task_record(flowid, task_id):
        Simulator._task_record_dict[task_id].remove(flowid)

    @staticmethod
    def is_task_done(task_id):
        if len(Simulator._task_record_dict[task_id]) == 0:
            return True
        else:
            return False

    @staticmethod
    def is_in_the_same_server(NIC_src, NIC_dst):
        NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        if NIC_src // NIC_num_in_a_server == NIC_dst // NIC_num_in_a_server:
            return True
        else:
            return False

    @staticmethod
    def get_wait_transmit_dict() -> Dict[str, Dict[int, Flow]]:
        """
        {'taskid_roundid': {'flowid', Flow}, ...}
        """
        return Simulator._wait_transmit_dict

    @staticmethod
    def get_finish_transmit_dict():
        return Simulator._finish_transmit_dict

    @staticmethod
    def value_in_wait_transmit_dict_is_empty():
        flag = True
        for k, v in Simulator._wait_transmit_dict.items():
            if len(v) > 0:
                flag = False
        return flag

    @staticmethod
    def add_a_wait_transmit_flow(taskid, roundid, flow: Flow):
        """
        {'taskid_roundid': {'flowid', Flow}, ...}
        """
        flowid = flow.get_flow_id()
        if Simulator._wait_transmit_dict.get(f'{taskid}_{roundid}') is None:
            Simulator._wait_transmit_dict[f'{taskid}_{roundid}'] = {}
        Simulator._wait_transmit_dict[f'{taskid}_{roundid}'][flowid] = flow

    @staticmethod
    def del_a_wait_transmit_flow(taskid, roundid, flowid):
        Simulator._wait_transmit_dict[f'{taskid}_{roundid}'].pop(flowid)

    @staticmethod
    def get_final_roundid(taskid):
        key_list = Simulator._wait_transmit_dict.keys()
        roundid_list = []
        for key in key_list:
            if f'{taskid}_' in key:
                roundid_list.append(int(key.split('_')[1]))
        return max(roundid_list)

    @staticmethod
    def load_scheduler():
        NIC_num = int(Simulator.CONF_DICT['NIC_num'])
        spine_switch_num = int(Simulator.CONF_DICT['spine_switch_num'])
        spine_switch_port_num = int(Simulator.CONF_DICT['spine_switch_port_num'])
        leaf_switch_num = int(Simulator.CONF_DICT['leaf_switch_num'])
        leaf_switch_port_num = int(Simulator.CONF_DICT['leaf_switch_port_num'])
        gpu_per_server = int(Simulator._global_conf.get_parameter('NIC_num_in_a_server'))

        # static means that do not reconfigure topology.
        # static_scheduler means scheduler based on switches' subClos.
        match Simulator.CONF_DICT['joint_scheduler']:
            case x if x in ['GPUPlacementer', 'GPUPlacementer3']:
                from rapidAIsim.scheduler import GpuPlacementer
                Simulator._scheduler = GpuPlacementer(spine_switch_num, leaf_switch_num, spine_switch_port_num,
                                                      leaf_switch_port_num,
                                                      int(spine_switch_num * spine_switch_port_num / gpu_per_server),
                                                      int(leaf_switch_port_num / 2))
            case 'GPUPlacementer2':
                from rapidAIsim.scheduler import GpuPlacementerRelax
                Simulator._scheduler = GpuPlacementerRelax(spine_switch_num, leaf_switch_num, spine_switch_port_num,
                                                           leaf_switch_port_num,
                                                           int(spine_switch_num * spine_switch_port_num / gpu_per_server),
                                                           int(leaf_switch_port_num / 2))
            case 'StaticPlacementer':
                from rapidAIsim.scheduler import StaticPlacementer
                Simulator._scheduler = StaticPlacementer(spine_switch_num, leaf_switch_num, spine_switch_port_num,
                                                         leaf_switch_port_num,
                                                         int(spine_switch_num * spine_switch_port_num / gpu_per_server),
                                                         int(leaf_switch_port_num / 2))
            case 'StaticPlacementerRelax':
                from rapidAIsim.scheduler import StaticPlacementerRelax
                Simulator._scheduler = StaticPlacementerRelax(
                    spine_switch_num, leaf_switch_num, spine_switch_port_num, leaf_switch_port_num,
                    int(spine_switch_num * spine_switch_port_num / gpu_per_server), int(leaf_switch_port_num / 2))
            case 'StaticPlacementerAI':
                from rapidAIsim.scheduler import StaticPlacementerAI
                root_path = Simulator.CONF_DICT['helper_root_path']
                with_link_weight = Simulator.CONF_DICT['with_link_weight'] == 'True'
                try:
                    model_path = Simulator.CONF_DICT['model_path']
                    if model_path == 'None':
                        model_path = None
                except KeyError:
                    model_path = None
                print("debug with_link_weight", with_link_weight)
                if Simulator.CONF_DICT['use_lstm'] == 'True':
                    from rapidAIsim.scheduler.static_locality_AI.leaf_spine_link_selector import LSTMInference
                    lstm_hidden_dim = int(Simulator.CONF_DICT['lstm_hidden_dim'])
                    lstm_layers = int(Simulator.CONF_DICT['lstm_layers'])
                    inference_model = LSTMInference(leaf_switch_num, spine_switch_num, root_path, with_link_weight,
                                                    lstm_hidden_dim, lstm_layers, model_path)
                else:
                    from rapidAIsim.scheduler.static_locality_AI.leaf_spine_link_selector import Inference
                    inference_model = Inference(leaf_switch_num, spine_switch_num, root_path, with_link_weight,
                                                model_path)
                Simulator._scheduler = StaticPlacementerAI(spine_switch_num, leaf_switch_num, spine_switch_port_num,
                                                           leaf_switch_port_num,
                                                           int(spine_switch_num * spine_switch_port_num / gpu_per_server),
                                                           int(leaf_switch_port_num / 2), inference_model)
            case 'GPUPlacementer4':
                from rapidAIsim.scheduler import GpuPlacementer4
                Simulator._scheduler = GpuPlacementer4(spine_switch_num, leaf_switch_num, spine_switch_port_num,
                                                       leaf_switch_port_num,
                                                       int(spine_switch_num * spine_switch_port_num / gpu_per_server),
                                                       int(leaf_switch_port_num / 2))
            case 'OCSExpander':
                from rapidAIsim.scheduler import OCSExpander
                ocs_reconfiguration = Simulator.CONF_DICT['ocs_reconfiguration'] == 'yes'
                Simulator._scheduler = OCSExpander(ocs_reconfiguration=ocs_reconfiguration)
                print("OCS reconfiguration is", ocs_reconfiguration)
                if ocs_reconfiguration is False:
                    Simulator._scheduler.init_mesh_topo()
                    connection_info_list = Simulator._scheduler.allocated_link_mapping
                    Simulator._infra_base.create_topology(connection_info_list)

                # Figret integration
                print("Figret integration is", Simulator.CONF_DICT['figret_integration'] == 'yes')
                if Simulator.CONF_DICT['figret_integration'] == 'yes':
                    Simulator.init_figret()

                # FigretToE integration
                print("FigretToE integration is", Simulator.CONF_DICT['figret_toe_integration'] == 'yes')
                if Simulator.CONF_DICT['figret_toe_integration'] == 'yes':
                    Simulator.init_figret_toe()
                    
            case 'OCSExpander_superPod':
                from rapidAIsim.scheduler import OCSExpander_superPod
                ocs_reconfiguration = Simulator.CONF_DICT['ocs_reconfiguration'] == 'yes'
                Simulator._scheduler = OCSExpander_superPod(ocs_reconfiguration=ocs_reconfiguration)
                print("OCS reconfiguration is", ocs_reconfiguration)
                if ocs_reconfiguration is False:
                    Simulator._scheduler.init_mesh_topo()
                    connection_info_list = Simulator._scheduler.allocated_link_mapping
                    Simulator._infra_base.create_topology(connection_info_list)

                # Figret integration
                print("Figret integration is", Simulator.CONF_DICT['figret_integration'] == 'yes')
                if Simulator.CONF_DICT['figret_integration'] == 'yes':
                    Simulator.init_figret()

                # FigretToE integration
                print("FigretToE integration is", Simulator.CONF_DICT['figret_toe_integration'] == 'yes')
                if Simulator.CONF_DICT['figret_toe_integration'] == 'yes':
                    Simulator.init_figret_toe()


            case 'ELEExpander':
                from rapidAIsim.scheduler import ELEExpander
                Simulator._scheduler = ELEExpander()
            
            case 'ELEExpander_SuperPod':
                from rapidAIsim.scheduler import ELEExpander_SuperPod
                Simulator._scheduler = ELEExpander_SuperPod()

            case x if x in ['static_scheduler', 'static']:
                from rapidAIsim.scheduler import StaticScheduler
                Simulator._scheduler = StaticScheduler(spine_switch_num, leaf_switch_num, False,
                                                       int(spine_switch_num * spine_switch_port_num / gpu_per_server))
            case x if x in ['static_scheduler_small']:
                from rapidAIsim.scheduler import static_small
                Simulator._scheduler = static_small(spine_switch_num, leaf_switch_num, False,
                                                    int(spine_switch_num * spine_switch_port_num / gpu_per_server))
            case x if x in ['hw_oxc_all2all', 'hw_oxc_all2all_sz', 'hw_oxc_all2all2',
                            'hw_oxc_allreduce', 'hw_oxc_hdallreduce', 'hw_eps_all2all',
                            'hw_eps_all2all_old', 'hw_eps_all2all2', 'hw_eps_all2all_hierarchical']:
                from rapidAIsim.scheduler import HwOxcAll2allScheduler
                NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
                Simulator._scheduler = HwOxcAll2allScheduler(NIC_num_in_a_server, NIC_num)
            case 'NaiveScheduler':
                from rapidAIsim.scheduler import NaiveScheduler
                # tor_num = int(Simulator.CONF_DICT['tor_num'])
                # ele_group_num = int(Simulator.CONF_DICT['ele_group_num'])
                # oxc_group_num = int(Simulator.CONF_DICT['oxc_group_num'])
                Simulator._scheduler = NaiveScheduler(4096, 8, 4)
            case 'NaiveSchedulerCompare':
                from rapidAIsim.scheduler import NaiveSchedulerCompare
                # tor_num = int(Simulator.CONF_DICT['tor_num'])
                # ele_group_num = int(Simulator.CONF_DICT['ele_group_num'])
                # oxc_group_num = int(Simulator.CONF_DICT['oxc_group_num'])
                Simulator._scheduler = NaiveSchedulerCompare(4096, 4)

    @staticmethod
    def init_figret():
        from figret.src import Figret, FigretEnv, FigretNetWork
        import torch
        config = Simulator.CONF_DICT
        props = argparse.Namespace()
        setattr(props, 'topo_name', config['figret_topo_name'])
        setattr(props, 'paths_file', 'tunnels.txt')
        setattr(props, 'data_dir', config['figret_data_dir'])
        setattr(props, 'dataset_label', None)
        model_path = config['figret_model_path']
        beta = float(model_path.split('_')[-1][:-3])
        setattr(props, 'beta', beta)
        env = FigretEnv(props)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Reading figret model from", model_path)
        model_name = os.path.basename(model_path)
        model_name = model_name.split('_')
        hist_len = int(model_name[1])
        num_layer = int(model_name[2])
        Simulator.figret = Figret(props, env, device)
        figret_model = FigretNetWork(hist_len * env.num_nodes * (env.num_nodes - 1), env.num_paths,
                                     num_layer).double()
        figret_model.load_state_dict(torch.load(model_path))
        figret_model.to(device)
        Simulator.figret.set_model(figret_model)

    @staticmethod
    def init_figret_toe():
        from figret_topo.src import FigretToE, FigretToEEnv, FigretNetWork
        import torch
        config = Simulator.CONF_DICT
        props = argparse.Namespace()
        setattr(props, 'topo_name', config['figret_topo_name'])
        setattr(props, 'paths_file', 'tunnels.txt')
        setattr(props, 'data_dir', config['figret_data_dir'])
        setattr(props, 'dataset_label', None)
        model_path = config['figret_toe_model_path']
        beta = float(model_path.split('_')[-2])
        gamma = float(model_path.split('_')[-1][:-3])
        setattr(props, 'beta', beta)
        setattr(props, 'gamma', gamma)
        pod_num = int(config['pod_num'])
        spine_num = int(config['spine_switch_num']) // int(config['pod_num'])
        setattr(props, 'pod_num', pod_num)
        setattr(props, 'spine_num_per_pod', spine_num)

        env = FigretToEEnv(props)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Reading figret ToE model from", model_path)
        model_name = os.path.basename(model_path)
        model_name = model_name.split('_')
        hist_len = int(model_name[1])
        layer_num = int(model_name[2])
        Simulator.figret_toe = FigretToE(props, env, device)
        figret_toe_model = FigretNetWork(hist_len, pod_num, spine_num, layer_num).double()
        figret_toe_model.load_state_dict(torch.load(model_path))
        figret_toe_model.to(device)
        Simulator.figret_toe.set_model(figret_toe_model)

    @staticmethod
    def get_scheduler():
        return Simulator._scheduler

    @staticmethod
    def init_logger():
        # open('./leaf_num_request.txt', 'w')
        # open('./original_state.txt', 'w')
        # open('./leaf_valid_vector.txt', 'w')
        # open('./link_weight_numpy.txt', 'w')
        # open('./problems.csv', 'w')
        open('./conflict_status.txt', 'w')
        open('./cross_pod_flow.log', 'w')

        # open('./link_uti.txt', 'w')
        # open('./task_info.txt', 'w')
        # Simulator.task_queue_length_logger = open('./cur_task_length.txt', 'w')
        # Simulator.task_queue_length_logger = open('./cur_gpu_num.txt', 'w')
        # open('./fixed_requests_256_part.txt', 'w')
        # open('./fragmentation.txt', 'w')
        with open('./fct.csv', 'w') as f:
            f.write('taskid,fct,exp_fct,start_time,end_time,flowid,src,dst,flow_size\n')
        Simulator.task_comm_ratio_logger = open('./task_comm_ratio.txt', 'w')
        # open('./spine_utility_timeline.txt', 'w')
        # open('./utility_timeline.txt', 'w')
        # open('./exp_running_time.txt', 'w')
        # open('./gpu_utilization.txt', 'w')
        # open('./job_usage_log.txt', 'w')
        # open('./debug_wcmp.txt', 'w')
        # 以下两个日志用于locality_scheduler
        # open('./schedule_time_cost.txt', 'w')
        # open('./schedule_time_cost_m*n.txt', 'w')
        # Simulator.task_queue_length_logger = open('./queue_length.txt', 'w')
        Simulator.task_time_logger = open('./task_time.log', 'w')
        # Measure event已经弃用，不再需要这个日志
        # Simulator.occupied_num_logger = open('./occupied_num.log', 'w')
        # Simulator.occupied_num_logger.write('time,src,dst,occupied_num,ave_bandwidth\n')
        Simulator.traffic_statistic_logger = open('./traffic_matrix.log', 'w')

    @staticmethod
    def get_logger(logger):
        match logger:
            # case 'task_queue_length':
            #     return Simulator.task_queue_length_logger
            case 'task_time':
                return Simulator.task_time_logger
            # case 'occupied_num':
            #     return Simulator.occupied_num_logger
            case 'traffic_statistic':
                return Simulator.traffic_statistic_logger

    @staticmethod
    def close_logger():
        Simulator.task_time_logger.close()
        # Simulator.occupied_num_logger.close()
        # Simulator.task_queue_length_logger.close()
        Simulator.traffic_statistic_logger.close()
        Simulator.task_comm_ratio_logger.close()

    @staticmethod
    def flush_logger():
        Simulator.task_time_logger.flush()
        # Simulator.occupied_num_logger.flush()
        # Simulator.task_queue_length_logger.flush()
        Simulator.traffic_statistic_logger.flush()
        Simulator.task_comm_ratio_logger.flush()

    @staticmethod
    def is_spine_switch(node_id):
        spine_switch_num = int(Simulator.CONF_DICT['spine_switch_num'])
        leaf_switch_num = int(Simulator.CONF_DICT['leaf_switch_num'])
        NIC_num = int(Simulator.CONF_DICT['NIC_num'])
        return (NIC_num + leaf_switch_num <=
                node_id <
                NIC_num + leaf_switch_num + spine_switch_num)

    @staticmethod
    def is_leaf_switch(node_id):
        leaf_switch_num = int(Simulator.CONF_DICT['leaf_switch_num'])
        NIC_num = int(Simulator.CONF_DICT['NIC_num'])
        return NIC_num <= node_id < NIC_num + leaf_switch_num

    @staticmethod
    def is_GPU(node_id):
        return node_id < int(Simulator.CONF_DICT['NIC_num'])

    @staticmethod
    def get_node_type(node_id):
        spine_switch_num = int(Simulator.CONF_DICT['spine_switch_num'])
        leaf_switch_num = int(Simulator.CONF_DICT['leaf_switch_num'])
        NIC_num = int(Simulator.CONF_DICT['NIC_num'])
        if node_id < NIC_num:
            return 'GPU'
        if NIC_num <= node_id < NIC_num + leaf_switch_num:
            return 'leaf'
        if NIC_num + leaf_switch_num <= node_id < NIC_num + leaf_switch_num + spine_switch_num:
            return 'spine'
        return 'core'

    @staticmethod
    def push_a_waiting_task(a_waiting_task):
        heapq.heappush(Simulator.WAITING_TASK_LIST, a_waiting_task)

    @staticmethod
    def pop_a_waiting_task():
        job_class_queuelength_map = {}
        for temp_task in Simulator.WAITING_TASK_LIST:
            if temp_task._task_occupied_NIC_num not in job_class_queuelength_map:
                job_class_queuelength_map[temp_task._task_occupied_NIC_num] = 0
            job_class_queuelength_map[temp_task._task_occupied_NIC_num] += 1
        for temp_task in Simulator.WAITING_TASK_LIST:
            temp_task._queue_length = job_class_queuelength_map[temp_task._task_occupied_NIC_num]
        for temp_task in Simulator.WAITING_TASK_LIST:
            if len(Simulator.WAITING_TASK_LIST) > 6:
                temp_task._weight_length = -temp_task._task_occupied_NIC_num
            else:
                temp_task._weight_length = -temp_task._taskid
        return heapq.heappop(Simulator.WAITING_TASK_LIST)

    @staticmethod
    def set_NIC_to_spine_map(nic_id, spine_id):
        Simulator.get_infrastructure().get_device(nic_id).set_to_spine_id(spine_id)

    @staticmethod
    def add_link_occupied_for_tasks(taskid, src, dst):
        if Simulator.LINK_OCCUPIED_FOR_TASKS.get((src, dst)):
            Simulator.LINK_OCCUPIED_FOR_TASKS[(src, dst)].append(taskid)
        else:
            Simulator.LINK_OCCUPIED_FOR_TASKS[(src, dst)] = [taskid]

    @staticmethod
    def del_link_occupied_for_tasks(taskid, src, dst):
        Simulator.LINK_OCCUPIED_FOR_TASKS[(src, dst)].remove(taskid)

    @staticmethod
    def get_hop_list_from_beforehand_path(src, dst):
        return Simulator.BEFOREHAND_PATH[(src, dst)]
