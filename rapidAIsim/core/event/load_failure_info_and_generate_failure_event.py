import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np
from rapidAIsim.core.event.FailureEvent import FailureEvent


def generate_failure_info(num, failure_interval=9000, soft_failure_recover_time=100, hard_failure_recover_time=500, soft_hard_ratio=0.1):
    # 根据软故障硬故障比例生成故障恢复事件，软硬故障比例为0.1，软故障恢复时间100，硬故障恢复时间500
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(failure_interval, num)
    f3 = open('failure_info.txt', 'w')
    # 根据软硬故障比例生成故障信息(故障id，发送时间，恢复时间)
    occur_time = 0
    for i in range(num):
        occur_time += exponential_interval_list[i]
        random.seed(i)
        if random.uniform(0, 1) < soft_hard_ratio:
            repair_time = soft_failure_recover_time
        else:
            repair_time = hard_failure_recover_time
        f3 = open('failure_info.txt','a')
        f3.write(str(i))
        f3.write(',')
        f3.write(str(occur_time))
        f3.write(',')
        f3.write(str(repair_time))
        f3.write('\n')


def load_failure_info_and_generate_failure_event(file_path='xxx/rapidAIsim/core/event/failure_info.txt'):
    from rapidAIsim.core.simulator import Simulator
    with open(file_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            failure_id = row[0]
            occur_time = row[1]
            duration_time = row[2]
           
            Simulator.register_event(
                FailureEvent(
                    float(occur_time),
                    int(failure_id),
                    float(duration_time),
                )
            )


def generate_failure_event(num, failure_interval=9000, soft_failure_recover_time=100, hard_failure_recover_time=500, soft_hard_ratio=0.1):
    """
    根据软故障硬故障比例生成故障恢复事件，软硬故障比例为0.1，软故障恢复时间100，硬故障恢复时间500
    """
    from rapidAIsim.core.simulator import Simulator
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(failure_interval, num)
    f0 = open('failure_info.txt','w')
    # 根据软硬故障比例生成故障信息(故障id，发送时间，恢复时间)
    occur_time = 0
    for i in range(num):
        occur_time += exponential_interval_list[i]
        random.seed(i)
        if random.uniform(0, 1) < soft_hard_ratio:
            repair_time = soft_failure_recover_time
        else:
            repair_time = hard_failure_recover_time
        if occur_time < 977245:
            Simulator.register_event(
                FailureEvent(
                    float(occur_time),
                    i,
                    float(repair_time),
                )
            )


def generate_link_failure_event(num, failure_interval=9000, repair_time=6000):
    # 根据软故障硬故障比例生成故障恢复事件，软硬故障比例为0.1，软故障恢复时间100，硬故障恢复时间500
    from rapidAIsim.core.simulator import Simulator
    from rapidAIsim.core.event.LinkFailureEvent import LinkFailureEvent
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(failure_interval, num)
    # 根据软硬故障比例生成故障信息(故障id，发送时间，恢复时间)
    occur_time = 0
    for i in range(num):
        occur_time += exponential_interval_list[i]
        random.seed(i + int(repair_time))
        if occur_time < 977245:
            Simulator.register_event(
                LinkFailureEvent(
                    float(occur_time),
                    i,
                    float(repair_time),
                )
            )
            
