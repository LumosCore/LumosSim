from dataclasses import dataclass


@dataclass
class TaskInfo:
    task_id: int
    arriving_time: float
    gpu_num: int
    computation_time: float = -1.0
    computation_round: int = 1
    model_size: float = -1.0
    duration_time: float = -1.0
    task_type: str = ''
    model_type: str = ''
    task_iteration_num: int = 10
    TP: int = 8
    DP: int = -1
    PP: int = -1
    EP: int = -1
