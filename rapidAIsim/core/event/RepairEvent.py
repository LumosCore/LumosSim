from rapidAIsim.core.event.event import Event


class RepairEvent(Event):
    def __init__(self, time_flow_now, failure_id) -> None:
        super().__init__(time_flow_now)
        self.time_flow_now = time_flow_now
        self.failure_id = failure_id
        self._type_priority = 2

    def __str__(self) -> str:
        print_dict = {
            'event_time': self.time_flow_now, 
            'failure_id': self.failure_id,
        }
        print_str = '<RepairEvent | '
        for key, val in print_dict.items():
            print_str += key + ': ' + str(val) + ', '
        print_str += '>'
        return print_str

    def do_sth(self):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['joint_scheduler'] in ['ELEExpander','OCSExpander_superPod', 'OCSExpander','ELEExpander_SuperPod']:
            Simulator._scheduler.handle_repair_event(self.failure_id)
