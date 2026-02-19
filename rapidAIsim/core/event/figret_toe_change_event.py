from .event import Event
from rapidAIsim.utils.traffic_matrix_helper import get_traffic_matrix
import numpy as np
from copy import deepcopy


class FigretToEChangeEvent(Event):
    def __init__(self, relative_time_from_now, interval):
        """
        :param relative_time_from_now 表示这个事件发生的时间间隔。
        """
        super().__init__(relative_time_from_now, interval)
        self.relative_time_from_now = relative_time_from_now
        self.event_interval = interval
        self._type_priority = 3

    def do_sth(self):
        from rapidAIsim.core.simulator import Simulator
        hist_len = int(Simulator.CONF_DICT['figret_hist_len'])
        traffic_matrix = get_traffic_matrix(self.event_time)
        Simulator.history_traffic_matrix.append(traffic_matrix)
        if len(Simulator.history_traffic_matrix) == hist_len:
            traffic_matrix = np.array(Simulator.history_traffic_matrix)
            Simulator.history_traffic_matrix.pop(0)
            # change_weight
            self.calculate_and_change_capacity(traffic_matrix)
        # register next event
        if len(Simulator._event_q) != 0:
            # avoid registering the event when the simulator is shutting down
            Simulator.register_event(FigretToEChangeEvent(self.relative_time_from_now, self.relative_time_from_now))

    def calculate_and_change_capacity(self, traffic_matrices):
        """
        根据流量矩阵计算新的链路带宽，并改变链路带宽。
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.scheduler.ocsexpander.TE_solver_lp import solve
        infra = Simulator.get_infrastructure()
        spine_up_port_num = infra.spine_switch_port_num // 2
        capacities = Simulator.figret_toe.inference(traffic_matrices, infra.spine_switch_num)
        c_ijt = []
        for i in range(spine_up_port_num):
            T_a_b = np.zeros((infra.pod_num, infra.pod_num))
            for j in range(infra.pod_num):
                T_a_b[j, :j] = capacities[j, i, :j]
                T_a_b[j, j + 1:] = capacities[j, i, j:]
            c_ij = solve(infra.pod_num, infra.spine_switch_num // infra.pod_num, spine_up_port_num, T_a_b)
            c_ijt.append(c_ij)
        c_ijt = np.stack(c_ijt)
        scheduler = Simulator.get_scheduler()
        x_ijkt = scheduler.generate_ocs_configuration(c_ijt)
        scheduler.u_ijkt = deepcopy(x_ijkt)
        allocated_link_mapping = scheduler.translate_link_new(scheduler.u_ijkt)
        scheduler.allocated_link_mapping = allocated_link_mapping
        Simulator.reconfigure(allocated_link_mapping, -2)
