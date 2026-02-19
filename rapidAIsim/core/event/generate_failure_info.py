import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math


def generate_failure_info(num, failure_interval=9000, soft_failure_recover_time=100, hard_failure_recover_time=500, soft_hard_ratio=0.1):
    # 根据软故障硬故障比例生成故障恢复事件，软硬故障比例为0.1，软故障恢复时间100，硬故障恢复时间500
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(failure_interval, num)
    open('failure_info.txt','w')
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


generate_failure_info(1000)