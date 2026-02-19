from rapidAIsim.core.simulator import Simulator


class WaitingTask:
    def __init__(self, arriving_time, model_size, task_occupied_NIC_num, task_type_obj, taskid, _exp_run_time) -> None:
        self._arriving_time = arriving_time
        self._model_size = model_size
        self._task_occupied_NIC_num = task_occupied_NIC_num
        self._task_type_obj = task_type_obj
        self._taskid = taskid
        self._queue_length = -1
        self._weight_length = -1
        self._exp_run_time = _exp_run_time

    def __lt__(self, other):
        waiting_task_order_mode = Simulator.CONF_DICT['waiting_task_order_mode']
        if waiting_task_order_mode == 'FIFO':
            return self._taskid < other._taskid
        elif waiting_task_order_mode == 'few_GPU_first':
            return self._task_occupied_NIC_num < other._task_occupied_NIC_num
        elif waiting_task_order_mode == 'earliest_first':
            return self._exp_run_time < other._exp_run_time
        elif waiting_task_order_mode == 'small_task_first_few_GPU_first':
            return self._task_occupied_NIC_num < other._task_occupied_NIC_num or (self._task_occupied_NIC_num == other._task_occupied_NIC_num and self._exp_run_time < other._exp_run_time)
        elif waiting_task_order_mode == 'max_weight_matching':
            return self._queue_length > other._queue_length
        elif waiting_task_order_mode == 'dynamic_matching':
            return self._weight_length > other._weight_length
        else:
            raise Exception('The waiting_task_order_mode does not exist!')

    def get_task_info(self):
        arriving_time = self._arriving_time
        model_size = self._model_size
        task_occupied_NIC_num = self._task_occupied_NIC_num
        task_type_obj = self._task_type_obj
        taskid = self._taskid
        return arriving_time, model_size, task_occupied_NIC_num, task_type_obj, taskid
