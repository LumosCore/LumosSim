
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

class Independent(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list):
        """When comparing with Netbench, one flow corresponds to one task.
        """
        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

        flow_arrivals_list = eval(Simulator.CONF_DICT['flow_arrivals_list'])

        (start_time_ns, src, dst, integer_byte) = flow_arrivals_list[taskid]
        model_size = integer_byte

        flow_list = []
        flow = Flow(Simulator.FLOWID, model_size, None, src, dst, model_size, None, taskid, 0, task_occupied_NIC_num,
                    False)
        self.record_network_occupy(taskid, 0, flow, src)
        flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(0, flow_list))
        Simulator.FLOWID += 1
