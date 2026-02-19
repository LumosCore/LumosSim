from typing import List
from rapidAIsim.core.event.event import Event
from rapidAIsim.core.infrastructure.flow import Flow
from collections import defaultdict


class FlowCompletionEvent(Event):
    """
    """
    def __init__(self, time_from_now, flows: List[Flow]) -> None:
        super().__init__(time_from_now)
        self._flows = flows
        self._type_priority = 0
        from rapidAIsim.core.simulator import Simulator
        self.event_init_time = Simulator.get_current_time()

    def __str__(self) -> str:
        print_dict = {
            'event_time': self.event_time,
            'flows': self._flows,
        }
        print_str = '<FlowCompletionEvent | '
        for key, val in print_dict.items():
            print_str += key + ': ' + str(val) + ', '
        print_str += '>'
        return print_str

    @property
    def flows(self):
        return self._flows

    def do_sth(self):
        # print('debug: FlowCompletionEvent.event_init_time', self.event_init_time)
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.stage_controller import del_global_record_trigger_new_round
        infra = Simulator.get_infrastructure()
        conservative = Simulator.CONF_DICT['find_next_hop_method'] == 'conservative'

        task_flow_map = defaultdict(set)
        task_round_map = {}
        for flow in self._flows:
            src = flow.get_src()
            dst = flow.get_dst()
            round_id = flow.get_round_id()
            flow_id = flow.get_flow_id()
            task_id = flow.get_task_id()
            hop_list = flow.get_hop_list()
            flow_start_time = flow.get_start_time()
            total_round = flow.get_total_round()
            flow_size = flow.get_size()
            task_flow_map[task_id].add(flow_id)
            task_round_map[task_id] = round_id
            Simulator.task_has_communication_size[task_id] += flow.get_size()
            Simulator.task_actual_comm_time[task_id] += self.event_time - flow_start_time
            # Release link resources
            # Start subsequent paths.
            tmp_src = src
            for next_hop in hop_list:
                # Ongoing path is (tmp_src, next_hop)
                # Necessary: Delete network-occupied flow.
                infra.del_link_flow_occupy(flow_id, tmp_src, next_hop, task_id)
                if conservative is True:
                    Simulator.del_link_occupied_for_tasks(task_id, tmp_src, next_hop)
                # Update next hop path
                tmp_src = next_hop

            # Record flow into traffic matrix
            if Simulator.CONF_DICT['traffic_matrix_statistics'] == 'yes':
                record_size = flow.traffic_matrix_remainder_size
                interAS_hop_list = flow.get_interAS_hop_list()
                for (src, dst, index) in interAS_hop_list:
                    traffic_matrix = Simulator.get_traffic_matrix()
                    traffic_matrix[src, dst, index] += record_size

            Simulator.GPU_status[src] = 0
            Simulator.GPU_status[dst] = 0
            Simulator.del_flowid_from_task_record(flow_id, task_id)
            Simulator.del_a_wait_transmit_flow(task_id, round_id, flow_id)
            gpu_per_pod = int(Simulator.CONF_DICT['server_num_per_pod'])*int(Simulator.CONF_DICT['NIC_num_in_a_server'])
            exp_finish_time = flow_size/int(Simulator.CONF_DICT['switch_port_bandwidth']) * total_round
            if round_id >= total_round - 1:
                with open('./fct.csv', mode='a') as f:
                    f.write(f'{task_id},{Simulator.get_current_time() - flow_start_time},'
                            f'{exp_finish_time},{flow_start_time},{Simulator.get_current_time()},'
                            f'{flow_id},{src},{dst},{src//gpu_per_pod},{dst//gpu_per_pod},{flow_size}\n')

        for task_id, flow_ids in task_flow_map.items():
            infra.del_flows_infly_info(flow_ids, task_id)
            round_id = task_round_map[task_id]
            del_global_record_trigger_new_round(task_id, round_id)
