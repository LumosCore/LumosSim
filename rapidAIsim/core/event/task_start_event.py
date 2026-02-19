from rapidAIsim.core.event.event import Event
from rapidAIsim.task.waiting_task import WaitingTask
import rapidAIsim.core.stage_controller as sc
import math


def is_power_of_2(n):
    return n & (n - 1) == 0


class TaskStartEvent(Event):
    """Trigger flows in task when simulation time has reached arriving time.
    """

    def __init__(self, time_from_now, model_size, task_occupied_NIC_num, task_type_obj, taskid,
                 task_iteration_num) -> None:
        super().__init__(time_from_now)
        self._time_from_now = time_from_now
        self._model_size = model_size
        self._task_occupied_NIC_num = task_occupied_NIC_num
        self._task_type_obj = task_type_obj
        self._taskid = taskid
        self._type_priority = 2
        self._task_iteration_num = task_iteration_num

    def __repr__(self) -> str:
        print_dict = {
            'time_from_now': self._time_from_now,
            'event_time': self.event_time,
            'taskid': self._taskid,
        }
        print_str = '<TaskStartEvent | '
        for key, val in print_dict.items():
            print_str += key + ': ' + str(val) + ', '
        print_str += '>'
        return print_str

    def do_sth(self):
        from rapidAIsim.core.simulator import Simulator

        task_occupied_NIC_num = self._task_occupied_NIC_num
        model_size = self._model_size
        task_type_obj = self._task_type_obj
        # print("debug task_type_obj",self._taskid,task_type_obj)
        taskid = self._taskid
        Simulator.task_type_map[taskid] = task_type_obj
        computation_time = Simulator.TASK_LIST[taskid].computation_time
        NIC_num_in_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        temp_z = int(pow(2, int(math.log2(task_occupied_NIC_num))))
        if task_occupied_NIC_num > NIC_num_in_server:
            comm_time = 0
            communication_size = model_size / 2
            for i in range(int(math.log2(task_occupied_NIC_num))):
                if i < int(math.log2(NIC_num_in_server)):
                    comm_time += communication_size / 1000
                else:
                    comm_time += communication_size / 100
                communication_size /= 2
            exp_finish_time = self._task_iteration_num * (comm_time * 2 + computation_time)
            if temp_z != task_occupied_NIC_num:
                exp_finish_time += self._task_iteration_num * 2 * model_size / 100
        elif task_occupied_NIC_num > 1:
            comm_time = 0
            communication_size = model_size / 2
            for i in range(int(math.log2(task_occupied_NIC_num))):
                comm_time += communication_size / 1000
                communication_size /= 2
            if comm_time == 0:
                comm_time = model_size / 1000 / 2
            exp_finish_time = self._task_iteration_num * (comm_time * 2 + computation_time)
        else:
            comm_time = 0
            exp_finish_time = self._task_iteration_num * (comm_time * 2 + computation_time)

        if len(Simulator.WAITING_TASK_LIST) == 0:
            scheduler = Simulator.get_scheduler()

            allocate_succeed, use_NIC_list = sc.allocate_a_task(scheduler, model_size, task_occupied_NIC_num,
                                                                task_type_obj, taskid)
            if not allocate_succeed:
                # If GPU resources is not enough, push the task information to global WAITING_TASK_LIST
                arriving_time = self.event_time

                a_waiting_task = WaitingTask(arriving_time, model_size, task_occupied_NIC_num, task_type_obj, taskid,
                                             exp_finish_time)

                Simulator.push_a_waiting_task(a_waiting_task)
            else:
                NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
                task_iteration_num = int(Simulator.CONF_DICT['task_iteration_num'])
                sc.continue_record_more_iteration_if_need(taskid, task_occupied_NIC_num, model_size, task_type_obj,
                                                          task_iteration_num, NIC_num_in_a_server, use_NIC_list)
        else:
            arriving_time = self.event_time
            a_waiting_task = WaitingTask(arriving_time, model_size, task_occupied_NIC_num, task_type_obj, taskid,
                                         exp_finish_time)
            Simulator.push_a_waiting_task(a_waiting_task)

        return
