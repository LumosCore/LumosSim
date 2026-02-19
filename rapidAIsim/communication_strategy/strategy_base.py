from abc import abstractmethod


class StrategyBase:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
        """
        This function is triggered when `allocate_task` returns Ture, means the task will start to transmit flows.
        It is used to generate the first iterations' flows of the giving task.
        """

    def record_network_occupy(self, taskid, roundid, flow, src):
        from rapidAIsim.core.simulator import Simulator
        Simulator.add_a_wait_transmit_flow(taskid, roundid, flow)
        Simulator.add_flowid_into_task_record(Simulator.FLOWID, taskid)

    @abstractmethod
    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        """
        Returns the pair list of the task during one iteration. One iteration may have multiple rounds. Flows in the
        same round will start to transmit at the same time.
        """
