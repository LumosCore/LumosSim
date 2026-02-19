from abc import abstractmethod


class Event:
    """Event abstract class, which is triggered by Simulator.
    Must be overrided.
    """
    STATIC_EID = 0

    def __init__(self, relative_time_from_now) -> None:
        """Create event which will happen the given amount of nanoseconds later.
        """
        assert relative_time_from_now >= 0

        # let import codes into funciton to void the error of cross-reference
        from rapidAIsim.core.simulator import Simulator
        self._event_time = Simulator.get_plan_event_time(relative_time_from_now)
        self._eid = Event.STATIC_EID
        self._type_priority = 0
        Event.STATIC_EID += 1
        self._active = True

    def __lt__(self, other):
        # if self._event_time < other.event_time:
        #     return True
        # elif self._event_time == other.event_time:
        #     if self._type_priority < other.type_priority:
        #         return True
        #     elif self._eid < other.eid:
        #         return True
        #     else:
        #         return False
        # else:
        #     return False
        return (self._event_time, self._type_priority, self._eid) < (other.event_time, other.type_priority, other.eid)

    @abstractmethod
    def do_sth(self):
        pass

    @property
    def event_time(self):
        return self._event_time

    def change_to_inactive(self):
        self._active = False

    @property
    def is_active(self):
        return self._active

    @property
    def eid(self):
        return self._eid

    @property
    def type_priority(self):
        return self._type_priority
