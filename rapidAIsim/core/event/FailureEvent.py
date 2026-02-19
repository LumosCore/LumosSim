from rapidAIsim.core.event.event import Event
from rapidAIsim.core.stage_controller import _detect_and_trigger_a_task


class FailureEvent(Event):
    def __init__(self, time_flow_now, failure_id, duration_time) -> None:
        super().__init__(time_flow_now)
        self.time_flow_now = time_flow_now
        self.failure_id = failure_id
        self.duration_time = duration_time
        self._type_priority = 1

    def __str__(self) -> str:
        print_dict = {
            'event_time': self.time_flow_now, 
            'duration_time': self.duration_time,
        }
        print_str = '<FailureEvent | '
        for key, val in print_dict.items():
            print_str += key + ': ' + str(val) + ', '
        print_str += '>'
        return print_str

    def do_sth(self):
        from rapidAIsim.core.simulator import Simulator
        if Simulator.CONF_DICT['joint_scheduler'] in ['ELEExpander','OCSExpander_superPod', 'OCSExpander','ELEExpander_SuperPod']:
            Simulator._scheduler.handle_failure_event(self.failure_id, self.duration_time)
            _detect_and_trigger_a_task()
        
        
