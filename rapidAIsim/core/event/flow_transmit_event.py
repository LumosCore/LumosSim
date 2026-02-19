

from rapidAIsim.core.event.event import Event


class FlowTransmitEvent(Event):
    """Trigger all the flows in flow_list at the same time.
    """
    def __init__(self, time_from_now, flow_list) -> None:
        super().__init__(time_from_now)
        self.comp_time = time_from_now
        self._flow_list = flow_list
        self._type_priority = 1
        from rapidAIsim.core.simulator import Simulator
        self._task_init_time = Simulator.get_current_time()

    def do_sth(self):
        """Start all the flows in flow_list.
        """
        from rapidAIsim.core.simulator import Simulator
        task_id = self._flow_list[0].get_task_id()
        for flow in self._flow_list:
            src = flow.get_src()
            dst = flow.get_dst()
            # round_id = flow.get_round_id()
            flow_id = flow.get_flow_id()
                
            flow.find_hop_list()
            hop_list = flow.get_hop_list()

            flow.set_start_time(Simulator.get_current_time())
            flow.set_last_calculated_time(Simulator.get_current_time())

            infra = Simulator.get_infrastructure()
            # If NIC_src and NIC_dst belong to the same server, 
            # do not accupy links bandwidth.

            # Start subsequent paths.
            tmp_src = src

            for next_hop in hop_list:
                # Ongoing path is (tmp_src, next_hop)
                # Necessary: Refresh network occupied condition.
                infra.add_link_flow_occupy(flow_id, tmp_src, next_hop, task_id)
                # Update next hop path
                tmp_src = next_hop
            # print("debug set_flow_infly_info", roundid, src, dst,flow_id, Simulator.get_current_time())
            infra.set_flow_infly_info(flow_id, flow, task_id)  # Necessary
            
            Simulator.GPU_status[src] = 1
            Simulator.GPU_status[dst] = 1
            if 'rerouting' in Simulator.CONF_DICT and Simulator.CONF_DICT['rerouting'] == 'yes':
                Simulator.flow_whether_can_rerouting[flow_id] = True
            else:
                Simulator.flow_whether_can_rerouting[flow_id] = False
            # print(f'flow {flow_id} of task {task_id} finish comp with {self.comp_time} at '
            #       f'{Simulator.get_current_time()} from {self._task_init_time}')
            # traffic_matrix, is_none_zero = infra.get_link_matrix()
            # if is_none_zero:
            #     f3 = open('link_uti.txt','a')
            #     f3.write( str(Simulator.get_current_time()))
            #     f3.write("&")
            #     f3.write( str(traffic_matrix))
            #     f3.write("\n" )
            #     f3.close()
        Simulator.task_has_computation_time[task_id] += self.comp_time

    @property
    def flow_list(self):
        return self._flow_list
