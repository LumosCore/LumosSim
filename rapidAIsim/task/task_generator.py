from rapidAIsim.core.event.task_start_event import TaskStartEvent
from rapidAIsim.task.task_info import TaskInfo
from rapidAIsim.communication_strategy import *
import random


class Task:

    @staticmethod
    def is_power_of_2(n):
        return n & (n - 1) == 0

    def generate(self):
        from rapidAIsim.core.simulator import Simulator

        task_type = Simulator.CONF_DICT['task_type']
        print('Task communication strategy:', task_type)
        task_list = eval(Simulator.CONF_DICT['task_list'])
        if task_type == 'llm':
            for task_id, task in enumerate(task_list):
                arriving_time, duration, model_type, gpu_num, task_iteration_num, TP, PP, DP, EP = task
                Simulator.TASK_LIST.append(TaskInfo(
                    task_id, arriving_time, gpu_num,
                    duration_time=duration, task_type=task_type, model_type=model_type,
                    task_iteration_num=task_iteration_num,
                    TP=TP, DP=DP, PP=PP, EP=EP,
                ))
        elif len(task_list[0] == 3):
            for task_id, task in enumerate(task_list):
                arriving_time, model_size, gpu_num = task
                Simulator.TASK_LIST.append(TaskInfo(
                    task_id, arriving_time, gpu_num,
                    model_size=model_size, task_iteration_num=int(Simulator.CONF_DICT['task_iteration_num'])
                ))
        else:
            for task_id, task in enumerate(task_list):
                arriving_time, model_size, task_occupied_NIC_num, computation_time, task_iteration_num = task[:5]
                Simulator.TASK_LIST.append(TaskInfo(
                    task_id, arriving_time, task_occupied_NIC_num,
                    model_size=model_size,
                    computation_time=computation_time,
                    task_iteration_num=task_iteration_num
                ))

        # Generate numerous jobs through call TaskStartEvent.
        for task in Simulator.TASK_LIST:
            if len(task_list[0]) != 3 and (task_type == 'randomly' or task_type == 'small_randomly'):
                task_type_obj = self._generate_random_task_type(task.task_id, task_type, task.gpu_num)
            else:
                task_type_obj = self.generate_task_obj(task_type)
            Simulator.task_time_logger.write(f'task_id,{task.task_id},arriving_time,{task.arriving_time}\n')
            Simulator.register_event(
                TaskStartEvent(
                    task.arriving_time,
                    task.model_size,
                    task.gpu_num,
                    task_type_obj,
                    task.task_id,
                    task.task_iteration_num,
                )
            )

    def _generate_random_task_type(self, taskid, task_type, nic_num):
        from rapidAIsim.core.simulator import Simulator
        if task_type == 'randomly':
            random.seed(taskid)
            random_value = random.uniform(0, 1)
            if random_value < 0.5 and self.is_power_of_2(nic_num):
                # print("choose HD")
                task_type_obj = Butterfly2()
                Simulator.task_class[taskid] = 'hd'
            elif random_value < 0.7:
                # print("choose ring")
                task_type_obj = Ring()
                Simulator.task_class[taskid] = 'ring'
            elif random_value < 0.95:
                # print("choose all2all")
                task_type_obj = All2All()
                Simulator.task_class[taskid] = 'alltoall'
            else:
                # print("choose all2allv")
                task_type_obj = All2All(0)
                Simulator.task_class[taskid] = 'alltoallv'
        elif task_type == 'small_randomly':
            if taskid == 0:
                task_type_obj = Ring()
            elif taskid == 1:
                task_type_obj = Butterfly2()
            elif taskid == 2:
                task_type_obj = Butterfly2()
            elif taskid == 3:
                task_type_obj = All2All()
            else:
                random.seed(taskid)
                random_value = random.uniform(0, 1)
                if random_value < 0.5 and self.is_power_of_2(nic_num):
                    # print("choose HD")
                    task_type_obj = Butterfly2()
                elif random_value < 0.7:
                    # print("choose ring")
                    task_type_obj = Ring()
                else:
                    # print("choose all2all")
                    task_type_obj = All2All()
        else:
            raise ValueError(f'Unknown task type: {task_type}')
        return task_type_obj

    @staticmethod
    def generate_task_obj(task_type):
        match task_type:
            case 'all_to_all':
                task_type_obj = AllToAll()
            case 'hierarchical_all2all':
                task_type_obj = HierachicalAll2All()
            case 'ring':
                task_type_obj = Ring()
            case 'butterfly':
                task_type_obj = Butterfly()
            case 'butterfly2':
                task_type_obj = Butterfly2()
            case 'pairwise':
                task_type_obj = All2All()
            case 'butterfly2D':
                task_type_obj = Butterfly2D()
            case 'randomly':
                task_type_obj = Butterfly2()
            case 'small_randomly':
                task_type_obj = Butterfly2()
            case 'no_comm':
                task_type_obj = NoCommunication()
            case 'butterfly3':
                task_type_obj = Butterfly3()
            case 'mesh_cross':
                task_type_obj = MeshCross()
            case 'compare_with_netbench':
                task_type_obj = Independent()
            case 'hw_oxc_all2all':
                task_type_obj = HwOxcAll2All()
            case 'hw_oxc_all2all2':
                task_type_obj = HwOxcAll2All2()
            case 'hw_oxc_all2all_sz':
                task_type_obj = HwOxcAll2AllSz()
            case 'hw_eps_all2all_old':
                task_type_obj = AllToAll()
            case 'hw_eps_all2all_hierachical':
                task_type_obj = HierachicalAll2All()
            case 'hw_eps_all2all':
                task_type_obj = HwOxcAll2All()
            case 'hw_eps_all2all2':
                task_type_obj = HwOxcAll2All2()
            case 'hw_oxc_allreduce':
                task_type_obj = HwOxcAllreduce()
            case 'hw_oxc_hdallreduce':
                task_type_obj = HwOxcHdAllreduce()
            case 'hw_eps_hdallreduce':
                task_type_obj = HwOxcHdAllreduce()
            case 'llm':
                task_type_obj = LLM()
            case _:
                raise ValueError(f'Unknown task type: {task_type}')
        return task_type_obj
