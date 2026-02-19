from .event import Event
from rapidAIsim.utils.traffic_matrix_helper import get_traffic_matrix


class TrafficStatisticEvent(Event):
    def __init__(self, relative_time_from_now, time_interval):
        """
        :param time_interval 表示测量流量矩阵的时间间隔。
        """
        super().__init__(relative_time_from_now)
        self.time_interval = time_interval
        self._type_priority = 3

    def do_sth(self):
        from rapidAIsim.core.simulator import Simulator
        traffic_matrix = get_traffic_matrix(self.event_time)
        # record traffic matrix
        traffic_matrix_str = ' '.join(map(str, traffic_matrix.flatten()))
        Simulator.traffic_statistic_logger.write(f'{self.event_time},{traffic_matrix_str}\n')
        Simulator.traffic_statistic_logger.flush()
        # register next event
        if len(Simulator._event_q) != 0:
            # avoid registering the event when the simulator is shutting down
            Simulator.register_event(TrafficStatisticEvent(self.time_interval, self.time_interval))
