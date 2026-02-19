import random
import csv
import os
from datetime import datetime
import copy
from math import ceil, log2, log
import pandas as pd
import numpy as np
import math

VCLOS_MODEL_LIST = ['Bert', 'ResNet50', 'ResNet101', 'VGG16', 'Pangu', 'llama_7b', 'llama2_7b', 'llama2_13b', 'GPT2_13b']
LARGE_MODEL_LIST = ["Pangu", "llama2_7b", "llama_7b", "llama2_13b", "GPT2_13b"]
SMALL_MODEL_LIST = ["VGG16_4", "VGG16_8", "VGG16_16", "VGG16_32", "VGG16_64", "VGG16_96",
                    "ResNet50_16", "ResNet50_32", "ResNet50_64", "ResNet50_96", "ResNet50_128",
                    "ResNet101_16", "ResNet101_32", "ResNet101_64", "ResNet101_96",
                    "Bert_16", "Bert_32"]

ALL_REDUCE_COST = {"VGG16_4": 0.25, "VGG16_8": 0.23, "VGG16_16": 0.16, "VGG16_32": 0.1, "VGG16_64": 0.07,
                   "VGG16_96": 0.04,
                   "ResNet50_16": 0.04, "ResNet50_32": 0.05, "ResNet50_64": 0.04, "ResNet50_96": 0.03,
                   "ResNet50_128": 0.02,
                   "ResNet101_16": 0.05, "ResNet101_32": 0.05, "ResNet101_64": 0.04, "ResNet101_96": 0.02,
                   "Bert_16": 0.12, "Bert_32": 0.08,
                   "Pangu": 0.0737,
                   "llama_7b": 0.069, "llama2_7b": 0.011, "llama2_13b": 0.0017, "GPT2_13b": 0.047}
ALL2ALL = {
    "Pangu": 0.258,
    "llama_7b": 0.255, "llama2_7b": 0.052, "llama2_13b": 0.058, "GPT2_13b": 0.145
}
PP = {
    "Pangu": 0.0058,
    "llama_7b": 0.0077, "llama2_7b": 0.0047, "llama2_13b": 0.0034, "GPT2_13b": 0.0063
}
ALLGATHER = {
    "Pangu": 0.0158
}
ReduceScatter = {
    "Pangu": 0.0002
}
Algo_Type = {
    "ALL2ALL": [0],
    "ALLREDUCE": [3, 4]
}

module_dir = os.path.dirname(os.path.abspath(__file__))
CLUSTER_LOG_PATH = os.path.join(module_dir, 'cluster_log_old.csv')

def get_vclos_model_info():
    running_map = {}
    running_with_contention_map = {}

    running_map[('llama_7b',16)] = 520
    running_map[('llama2_7b',16)] = 1074
    running_map[('llama2_13b',16)] = 1689
    running_map[('Pangu',16)] = 278

    running_map[('llama_7b',32)] = 520
    running_map[('llama2_7b',32)] = 1074
    running_map[('llama2_13b',32)] = 1689
    running_map[('Pangu',32)] = 278

    running_map[('llama_7b',64)] = 575
    running_map[('llama2_7b',64)] = 1082
    running_map[('llama2_13b',64)] = 1717
    running_map[('GPT2_13b',64)] = 903
    running_map[('Pangu',64)] = 279

    running_map[('llama_7b',96)] = 602.13
    running_map[('llama2_7b',96)] = 1103.38
    running_map[('llama2_13b',96)] = 1763.47
    running_map[('GPT2_13b',96)] = 956.83
    running_map[('Pangu',96)] = 371.78

    running_map[('llama_7b',128)] = 611
    running_map[('llama2_7b',128)] = 1110
    running_map[('llama2_13b',128)] = 1782
    running_map[('GPT2_13b',128)] = 963
    running_map[('Pangu',128)] = 424

    running_map[('VGG16',128)] = 153.531218
    running_map[('VGG16',96)] = 152.1298174
    running_map[('VGG16',64)] = 150.7537688
    running_map[('VGG16',32)] = 149.775337
    running_map[('VGG16',16)] = 146.8428781

    running_map[('ResNet50',128)] = 113.7656428
    running_map[('ResNet50',96)] = 112.7395716
    running_map[('ResNet50',64)] = 111.6902457
    running_map[('ResNet50',32)] = 111.2347052
    running_map[('ResNet50',16)] = 108.5383502

    running_map[('ResNet101',128)] = 195.9503592
    running_map[('ResNet101',96)] = 196.3350785
    running_map[('ResNet101',64)] = 196.0784314
    running_map[('ResNet101',32)] = 189.3939394
    running_map[('ResNet101',16)] = 187.1490954

    running_map[('Bert',128)] = 538.2131324
    running_map[('Bert',96)] = 531.3496281
    running_map[('Bert',64)] = 524.2005941
    running_map[('Bert',32)] = 514.7563487
    running_map[('Bert',16)] = 492.0452682

    running_with_contention_map[('llama_7b',32)] = 643
    running_with_contention_map[('llama2_7b',32)] = 1173
    running_with_contention_map[('llama2_13b',32)] = 1815
    running_with_contention_map[('Pangu',32)] = 328

    running_with_contention_map[('llama_7b',96)] = 791.62
    running_with_contention_map[('llama2_7b',96)] = 1163.24
    running_with_contention_map[('llama2_13b',96)] = 1886.59
    running_with_contention_map[('GPT2_13b',96)] = 1160.3
    running_with_contention_map[('Pangu',96)] = 488.74

    running_with_contention_map[('VGG16',128)] = 167.5041876
    running_with_contention_map[('VGG16',96)] = 165.7458564
    running_with_contention_map[('VGG16',64)] = 162.7780792
    running_with_contention_map[('VGG16',32)] = 158.4786054
    running_with_contention_map[('VGG16',16)] = 141.1100659

    running_with_contention_map[('ResNet50',128)] = 114.6350783
    running_with_contention_map[('ResNet50',96)] = 114.2421935
    running_with_contention_map[('ResNet50',64)] = 114.1552511
    running_with_contention_map[('ResNet50',32)] = 113.4215501
    running_with_contention_map[('ResNet50',16)] = 109.7694841

    running_with_contention_map[('ResNet101',128)] = 195.4397394
    running_with_contention_map[('ResNet101',96)] = 196.4636542
    running_with_contention_map[('ResNet101',64)] = 197.3684211
    running_with_contention_map[('ResNet101',32)] = 194.0491591
    running_with_contention_map[('ResNet101',16)] = 189.3939394

    running_with_contention_map[('Bert',128)] = 437.4453193
    running_with_contention_map[('Bert',96)] = 433.9022274
    running_with_contention_map[('Bert',64)] = 430.9106579
    running_with_contention_map[('Bert',32)] = 423.9084358
    running_with_contention_map[('Bert',16)] = 411.8616145

    comm_all_ration_map = {}
    for model_name in VCLOS_MODEL_LIST:
        for beta in [16, 32, 64, 96, 128]:
            if (model_name, beta) in running_map:
                y_1 = running_with_contention_map[(model_name, beta)]
                y_0 = running_map[(model_name, beta)]
                comm_all_ration_map[(model_name, beta)] = (y_1 - y_0) / y_0

    return comm_all_ration_map


def generate_helios_tpuv4_map(helios_distribution=None, tpuv4_distribution=None):
    if helios_distribution is None:
        df = pd.read_csv(CLUSTER_LOG_PATH, encoding='utf-8-sig', skipinitialspace=True)
        filtered_df = df[
            (df.iloc[:, 7].apply(date_time_str_to_long) >= date_time_str_to_long("2020-06-01 00:00:00")) &
            (df.iloc[:, 3] > 0)
            ]
        npu_num_map = filtered_df.iloc[:, 3].apply(lambda x: pow(2, ceil(log2(x)))).value_counts().to_dict()
        total_num = filtered_df.shape[0]
        helios_distribution = [[key, value / total_num] for key, value in npu_num_map.items()]

    if tpuv4_distribution is None:
        tpu_list = [32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048] #32,
        dis = [0.29, 0.43, 0.605, 0.612, 0.729, 0.73, 0.849, 0.956, 0.993, 0.995, 1] #0.29, 
        tpuv4_distribution = [[tpu, d - dis[i - 1]] if i > 0 else [tpu_list[0], d] for i, (tpu, d) in
                              enumerate(zip(tpu_list, dis))]
    helios_distribution.sort(key=lambda x: x[0])
    tpuv4_distribution.sort(key=lambda x: x[0])
    # print()
    # print("helios_distribution")
    # print(helios_distribution,len(helios_distribution))
    # print()
    # print(tpuv4_distribution,len(tpuv4_distribution))

    # helios_distribution_list_copy = copy.deepcopy(helios_distribution)
    helios_distribution_map = {k: v for k, v in helios_distribution}
    ptr1 = 0
    ptr2 = 0
    helios_npu_size_map = {}
    while ptr1 < len(helios_distribution) and ptr2 < len(tpuv4_distribution):
        h_tpu, h_dis = helios_distribution[ptr1]
        t_tpu, t_dis = tpuv4_distribution[ptr2]
        if h_dis < t_dis:
            if h_tpu not in helios_npu_size_map:
                helios_npu_size_map[h_tpu] = {}
            assert t_tpu not in helios_npu_size_map[h_tpu]
            helios_npu_size_map[h_tpu][t_tpu] = h_dis
            tpuv4_distribution[ptr2][1] = t_dis - h_dis
            ptr1 += 1
        elif h_dis > t_dis:
            if h_tpu not in helios_npu_size_map:
                helios_npu_size_map[h_tpu] = {}
            assert t_tpu not in helios_npu_size_map[h_tpu]
            helios_npu_size_map[h_tpu][t_tpu] = t_dis
            helios_distribution[ptr1][1] = h_dis - t_dis
            ptr2 += 1
        else:
            ptr1 += 1
            ptr2 += 1
    # print("debug ptr", ptr1, ptr2, len(helios_distribution), len(tpuv4_distribution))
    # print()
    # print("result")
    # print(helios_npu_size_map)
    # print(helios_distribution)

    for npu in helios_npu_size_map:
        for to_map_index in helios_npu_size_map[npu]:
            helios_npu_size_map[npu][to_map_index] = \
                helios_npu_size_map[npu][to_map_index] / helios_distribution_map[npu]
    # print(helios_npu_size_map)
    return helios_npu_size_map


def is_power_of_2(n):
    return n & (n - 1) == 0


def date_time_str_to_long(input_date_time_string):
    if input_date_time_string == 'None':
        return 0
    return datetime.strptime(input_date_time_string, "%Y-%m-%d %H:%M:%S").timestamp()


def _modify_to_2exponent(task_occupied_NIC_num):
    exponent = log2(task_occupied_NIC_num)
    exponent = ceil(exponent)
    exponent = max(1, exponent)
    task_occupied_NIC_num = 2 ** exponent

    return task_occupied_NIC_num


def generate_tasks(num):
    request_sequence = eval(open("../../base_conf_template/request_sequence", "r").read())
    assert num <= len(request_sequence)
    task_info = request_sequence[:num]
    new_task_info = []
    arriving_time = 0
    for (_, model_size, task_occupied_NIC_num) in task_info:
        new_task_occupied_NIC_num = _modify_to_2exponent(task_occupied_NIC_num)
        new_task_info.append((arriving_time, model_size, new_task_occupied_NIC_num))
        # arriving_time += 0.1
    return new_task_info


def generate_custom_tasks(num, exponential_interval=True):
    new_task_info = []
    arriving_time = 0

    base_NIC_num_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_times_list = [3951, 782, 879, 2517, 669, 658, 243, 207, 94]

    job_weight_map = {}
    for job_class_id in base_NIC_num_list:
        job_weight_map[job_class_id] = job_class_id / sum(base_NIC_num_list)

    task_occupied_NIC_num_list = []
    arriving_time = 0
    for i in range(len(num_times_list)):
        for _ in range(num_times_list[i]):
            task_occupied_NIC_num_list.append(base_NIC_num_list[i])
    random.seed(0)
    random.shuffle(task_occupied_NIC_num_list)

    if exponential_interval == True:
        rng = np.random.default_rng(0)
        exponential_interval_list = rng.exponential(1, num)

    f1 = np.poly1d([2.28445089e+07, -5.04390049e+07, 3.92752625e+07, -1.27617787e+07, 1.55738356e+06, -4.34829130e+04])

    j = 0
    for _ in range(num):
        if exponential_interval == True:
            model_size = random.randint(1000, 10000)
            # model_size = max(10000,100*f1(random.uniform(0, 1)))
            arriving_time += exponential_interval_list[i]
        new_task_info.append((arriving_time, model_size, task_occupied_NIC_num_list[j]))
        f1 = open('fixed_requests_256_part.txt', 'a')
        f1.write(str(task_occupied_NIC_num_list[j]))
        f1.write(" ")
        f1.write(str(arriving_time))
        f1.write("\n")
        f1.close()
        j += 1
    # print('new_task_info:', new_task_info)
    return new_task_info


def get_exp_comm_time_HD(task_occupied_NIC_num, task_iteration_num=10, NIC_num_in_server=4):
    model_size = 1
    if task_occupied_NIC_num > NIC_num_in_server:
        comm_time = 0
        communication_size = model_size / 2
        for t in range(int(log2(task_occupied_NIC_num))):
            if t < int(log2(NIC_num_in_server)):
                comm_time += communication_size / 2000
            else:
                comm_time += communication_size / 1600
            communication_size /= 2
        exp_communication_time = task_iteration_num * (comm_time * 2)
    else:
        comm_time = 0
        communication_size = model_size / 2
        for t in range(int(log2(task_occupied_NIC_num))):
            comm_time += communication_size / 2000
            communication_size /= 2
        if comm_time == 0:
            comm_time = model_size / 1600 / 2
        exp_communication_time = task_iteration_num * (comm_time * 2)
    return exp_communication_time


def get_exp_comm_time_Ring(task_occupied_NIC_num, task_iteration_num=10, NIC_num_in_a_server=4):
    model_size = 1
    node_num = int(task_occupied_NIC_num / NIC_num_in_a_server)
    comm_time = 0

    if task_occupied_NIC_num > NIC_num_in_a_server:
        # ring allreduce in the intra-server
        round_num = NIC_num_in_a_server - 1
        communication_size = model_size / NIC_num_in_a_server
        for _ in range(round_num):
            comm_time += communication_size / 2000

        # ring allreduce inter servers
        communication_size = model_size / NIC_num_in_a_server / node_num
        round_num = node_num
        for _ in range(round_num):
            comm_time += communication_size / 1600

        # ring allreduce in the intra-server
        communication_size = model_size / NIC_num_in_a_server
        round_num = NIC_num_in_a_server - 1
        for _ in range(round_num):
            comm_time += communication_size / 2000
    else:
        round_num = task_occupied_NIC_num - 1
        communication_size = model_size / task_occupied_NIC_num
        for _ in range(round_num):
            comm_time += communication_size / 2000

    return comm_time


def get_exp_comm_time_all2all(task_occupied_NIC_num, task_iteration_num=10, NIC_num_in_a_server=4):
    model_size = 1
    comm_time = 0

    if task_occupied_NIC_num > NIC_num_in_a_server:
        round_num = task_occupied_NIC_num - 1
        communication_size = model_size
        for _ in range(round_num):
            comm_time += communication_size / 1600
    else:
        round_num = task_occupied_NIC_num - 1
        communication_size = model_size
        for _ in range(round_num):
            comm_time += communication_size / 2000
    return comm_time


def random_choose_model_name(taskid, task_occupied_NIC_num):
    random.seed(taskid)
    random_value = random.uniform(0, 1)
    if random_value < 0 and is_power_of_2(task_occupied_NIC_num):
        model_name = random.choice(SMALL_MODEL_LIST)
        exp_comm_time = get_exp_comm_time_HD(task_occupied_NIC_num)
    elif random_value < 0:
        model_name = random.choice(SMALL_MODEL_LIST)
        exp_comm_time = get_exp_comm_time_Ring(task_occupied_NIC_num)
    else:
        tmp_list = copy.deepcopy(LARGE_MODEL_LIST)
        # tmp_list.extend(SMALL_MODEL_LIST)
        model_name = random.choice(tmp_list)
        exp_comm_time = get_exp_comm_time_all2all(task_occupied_NIC_num)
    # print("debug random_value" ,taskid ,random_value, model_name)
    return model_name, exp_comm_time


def get_fixed_requests_256_part_tasks(num, exponential_interval=True, beta=1,
                                      filename='../../../base_conf_template/fixed_requests_256_part.txt', modify=False,
                                      random_beta=False, beta_list=None, ave_comm_ratio=0.2):
    if beta_list is None:
        beta_list = []
    map_relation = generate_helios_tpuv4_map()
    print("fuck beta_list", beta_list)
    open('task_detail.txt','w')
    waiting_list = []
    running_list = []
    gpu_list = []
    arrive_list = []
    PP_list = []
    DP_list = []
    EP_list = []
    with open(CLUSTER_LOG_PATH, encoding='utf-8-sig') as f:
        start_time = 0
        index = 0
        for row in csv.reader(f, skipinitialspace=True):
            if (row[6] == 'COMPLETED' or row[6] == 'CANCELLED'or row[6] == 'FAILED' or row[6] == 'NODE_FAIL'or row[6] == 'TIMEOUT' ) and int(row[3])>0 and date_time_str_to_long((row[7])) >= date_time_str_to_long("2020-06-01 00:00:00"): #2020-08-15 00:51:57 # 2020-06-02 08:51:57
                if start_time == 0:
                    start_time = date_time_str_to_long((row[7]))   
                
                duration_time = date_time_str_to_long(row[9])-date_time_str_to_long(row[8])
                waiting_time = date_time_str_to_long(row[8])-date_time_str_to_long(row[7])
                running_list.append(duration_time)
                waiting_list.append(waiting_time)
                arrive_list.append(date_time_str_to_long((row[7]))-start_time)
                original_gpu_size = min(512, ceil(log2(int(row[3]))))
                to_map_npu_size = randomly_chosen_accord_to_coff(map_relation[pow(2, original_gpu_size)], index)
                to_map_npu_size = 2 ** (int(math.log2(to_map_npu_size)))
                to_map_npu_size = min(2048, to_map_npu_size)
                to_map_npu_size = max(1, to_map_npu_size//8)
                gpu_list.append(to_map_npu_size)
                
                random.seed(len(PP_list))
                PP_tmp_list = [4, 8, 16, 32]
                pp_size = random.choice(PP_tmp_list)
                PP_list.append(min(to_map_npu_size, pp_size))
                DP_list.append(round(to_map_npu_size / min(to_map_npu_size, pp_size)))

                ep_tmp_list = [8, 16, 32, 64]
                ep_size = random.choice(ep_tmp_list)
                EP_list.append(min(ep_size, to_map_npu_size))
            index += 1
    data_set_size = len(gpu_list)
    if exponential_interval and not random_beta:
        rng = np.random.default_rng(num)
        exponential_interval_list = rng.exponential(beta, num)
    elif random_beta:
        exponential_interval_list_list = []
        for temp_beta in beta_list:
            rng = np.random.default_rng(num)
            exponential_interval_list = rng.exponential(temp_beta, num)
            exponential_interval_list_list.append(exponential_interval_list)
    else:
        raise ValueError("Invalid parameter setting")
    new_task_info = []
    arriving_time = 0
    chosen_gpu_list = []
    comm_all_ration_list = []
    total_val_list = []
    for i in range(num):
        task_occupied_NIC_num = gpu_list[i]
        chosen_gpu_list.append(task_occupied_NIC_num)
        comm_all_ration = 0

        model_name, exp_communication_time = random_choose_model_name(i, task_occupied_NIC_num) #exp_communication_Dp_time exp_communication_pp_time exp_communication_ep_time
        if model_name in ALL_REDUCE_COST:
            comm_all_ration += ALL_REDUCE_COST[model_name] * random.uniform(0.8, 1.2)
        if model_name in ALLGATHER:
            comm_all_ration += ALLGATHER[model_name] * random.uniform(0.8, 1.2)
        if model_name in ReduceScatter:
            comm_all_ration += ReduceScatter[model_name] * random.uniform(0.8, 1.2)
        if model_name in ALL2ALL:
            comm_all_ration += ALL2ALL[model_name] * random.uniform(0.8, 1.2)

        total_val = max(3600*random.uniform(0.8, 1.1), running_list[i % data_set_size])
        total_val = min(100000*random.uniform(0.8, 1.1), total_val)
        total_val = 100 * max(1, int(total_val / 100))
        task_iteration_num = 10
        model_size = 0
        if task_occupied_NIC_num > 1:
            model_size = total_val * comm_all_ration / exp_communication_time ##pp_size ep_size dp_size
        computation_time = total_val * (1 - comm_all_ration) / task_iteration_num
        if task_occupied_NIC_num == 1:
            computation_time = total_val / task_iteration_num
        if not random_beta:
            arriving_time += exponential_interval_list[i]
        else:
            arriving_time += exponential_interval_list_list[int(i / 100) % (len(beta_list))][i]
        comm_all_ration_list.append(comm_all_ration)
        total_val_list.append(total_val)
        new_task_info.append(
            (float(arriving_time), model_size, task_occupied_NIC_num * 8, computation_time, task_iteration_num,
             8, PP_list[i], DP_list[i], EP_list[i], total_val))
    # print(comm_all_ration_list)
    # print(total_val_list)
    return new_task_info


def get_fixed_requests_large_llm_tasks(num, beta):
    map_relation = generate_helios_tpuv4_map()
    print("beta = {}".format(beta))

    def get_npu_size(ori_npu_num):
        npu_power = np.ceil(np.log2(min(2048, ori_npu_num)))
        npu_num = randomly_chosen_accord_to_coff(map_relation[2 ** npu_power], ori_npu_num)
        npu_num = 2 ** (int(np.log2(npu_num)))
        npu_num = min(2048, npu_num)
        npu_num = max(1, npu_num // 8)
        return npu_num

    task_list = pd.read_csv(CLUSTER_LOG_PATH, encoding='utf-8', skipinitialspace=True)
    task_list = task_list[task_list['state'] == 'COMPLETED']
    task_list = task_list[['gpu_num', 'start_time', 'duration']]
    task_list = task_list[task_list['duration'] > 600]
    task_list = task_list[task_list['gpu_num'] > 0]
    task_list['start_time'] = pd.to_datetime(task_list['start_time'])
    task_list = task_list[task_list['start_time'] >= pd.to_datetime("2020-06-01 00:00:00")]
    task_list = task_list.reset_index(drop=True)
    task_list = task_list.loc[:num - 1, :]
    task_list['npu_num'] = task_list['gpu_num'].apply(get_npu_size)
    task_list['model_type'] = task_list.apply(lambda x: random.choice(LARGE_MODEL_LIST), axis=1)
    task_list['task_iteration_num'] = 10
    task_list['TP'] = 8
    pp_list = [1, 2, 4, 8]
    task_list['PP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(pp_list)))
    task_list['DP'] = task_list['npu_num'] // task_list.apply(lambda x: min(x['npu_num'], x['PP']), axis=1)
    ep_list = [32, 64, 128, 256] #[8, 16, 32]
    # task_list['EP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(ep_list)))
    def calculate_ep(row):
        ep_value = min(row['npu_num'], random.choice(ep_list))
        return max(8,min(ep_value, 8 * row['DP']))

    task_list['EP'] = task_list.apply(calculate_ep, axis=1)
    print("debug EP_size")
    print(task_list['EP'])
    
    task_list['npu_num'] *= 8
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(beta, num)
    task_list['arriving_time'] = np.cumsum(exponential_interval_list)
    new_task_info = task_list[['arriving_time', 'duration', 'model_type', 'npu_num', 'task_iteration_num',
                               'TP', 'PP', 'DP', 'EP']]
    new_task_info = new_task_info.values.tolist()
    print(len(new_task_info))
    return new_task_info


def get_fixed_requests_256_part_llm_tasks(num, beta):
    map_relation = generate_helios_tpuv4_map()
    print("beta = {}".format(beta))

    def get_npu_size(ori_npu_num):
        npu_power = np.ceil(np.log2(min(2048, ori_npu_num)))
        npu_num = randomly_chosen_accord_to_coff(map_relation[2 ** npu_power], ori_npu_num)
        npu_num = 2 ** (int(np.log2(npu_num)))
        npu_num = min(2048, npu_num)
        npu_num = max(1, npu_num // 8)
        return npu_num

    task_list = pd.read_csv(CLUSTER_LOG_PATH, encoding='utf-8', skipinitialspace=True)
    task_list = task_list[task_list['state'] == 'COMPLETED']
    task_list = task_list[['gpu_num', 'start_time', 'duration']]
    task_list = task_list[task_list['duration'] > 600]
    task_list = task_list[task_list['gpu_num'] > 0]
    task_list['start_time'] = pd.to_datetime(task_list['start_time'])
    task_list = task_list[task_list['start_time'] >= pd.to_datetime("2020-06-01 00:00:00")]
    task_list = task_list.reset_index(drop=True)
    task_list = task_list.loc[:num - 1, :]
    task_list['npu_num'] = task_list['gpu_num'].apply(get_npu_size)
    task_list['model_type'] = task_list.apply(lambda x: random.choice(LARGE_MODEL_LIST), axis=1)
    task_list['task_iteration_num'] = 10
    task_list['TP'] = 8
    pp_list = [1, 2, 4, 8, 16, 32]
    task_list['PP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(pp_list)))
    task_list['DP'] = task_list['npu_num'] // task_list.apply(lambda x: min(x['npu_num'], x['PP']), axis=1)
    ep_list = [8, 16, 32] #[8, 16, 32]
    task_list['EP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(ep_list)))
    # # task_list['EP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(ep_list)))
    # def calculate_ep(row):
    #     ep_value = min(row['npu_num'], random.choice(ep_list))
    #     return max(8,min(ep_value, 8 * row['DP']))

    # task_list['EP'] = task_list.apply(calculate_ep, axis=1)
    task_list['npu_num'] *= 8
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(beta, num)
    task_list['arriving_time'] = np.cumsum(exponential_interval_list)
    new_task_info = task_list[['arriving_time', 'duration', 'model_type', 'npu_num', 'task_iteration_num',
                               'TP', 'PP', 'DP', 'EP']]
    new_task_info = new_task_info.values.tolist()
    print(len(new_task_info))
    return new_task_info

def get_fixed_requests_256_part_llm_tasks2(num, beta):
    map_relation = generate_helios_tpuv4_map()
    print("beta = {}".format(beta))

    def get_npu_size(ori_npu_num):
        npu_power = np.ceil(np.log2(min(2048, ori_npu_num)))
        npu_num = randomly_chosen_accord_to_coff(map_relation[2 ** npu_power], ori_npu_num)
        npu_num = 2 ** (int(np.log2(npu_num)))
        npu_num = min(2048, npu_num)
        npu_num = max(1, npu_num // 8)
        return npu_num

    task_list = pd.read_csv(CLUSTER_LOG_PATH, encoding='utf-8', skipinitialspace=True)
    task_list = task_list[task_list['state'] == 'COMPLETED']
    task_list = task_list[['gpu_num', 'start_time', 'duration']]
    task_list = task_list[task_list['duration'] > 600]
    task_list = task_list[task_list['gpu_num'] > 0]
    task_list['start_time'] = pd.to_datetime(task_list['start_time'])
    task_list = task_list[task_list['start_time'] >= pd.to_datetime("2020-06-01 00:00:00")]
    task_list = task_list.reset_index(drop=True)
    task_list = task_list.loc[:num - 1, :]
    task_list['npu_num'] = task_list['gpu_num'].apply(get_npu_size)
    task_list['model_type'] = task_list.apply(lambda x: random.choice(LARGE_MODEL_LIST), axis=1)
    task_list['task_iteration_num'] = 10
    task_list['TP'] = 8
    pp_list = [1, 2, 4]
    task_list['PP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(pp_list)))
    task_list['DP'] = task_list['npu_num'] // task_list.apply(lambda x: min(x['npu_num'], x['PP']), axis=1)
    ep_list = [8] #[8, 16, 32]
    task_list['EP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(ep_list)))
    # # task_list['EP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(ep_list)))
    # def calculate_ep(row):
    #     ep_value = min(row['npu_num'], random.choice(ep_list))
    #     return max(8,min(ep_value, 8 * row['DP']))

    # task_list['EP'] = task_list.apply(calculate_ep, axis=1)
    task_list['npu_num'] *= 8
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(beta, num)
    task_list['arriving_time'] = np.cumsum(exponential_interval_list)
    new_task_info = task_list[['arriving_time', 'duration', 'model_type', 'npu_num', 'task_iteration_num',
                               'TP', 'PP', 'DP', 'EP']]
    new_task_info = new_task_info.values.tolist()
    print(len(new_task_info))
    return new_task_info



def get_fixed_requests_256_part_vclos_tasks(num, beta):
    print("beta = {}".format(beta))
    random.seed(0)

    def get_npu_size(line):
        # print(line)
        if line['model_type'] in LARGE_MODEL_LIST:
            npu_num = random.choice([16, 32, 64, 96])
        else:
            npu_num = random.choice([16, 32, 64])
        npu_num = npu_num // 8
        return npu_num

    task_list = pd.read_csv(CLUSTER_LOG_PATH, encoding='utf-8', skipinitialspace=True)
    task_list = task_list[task_list['state'] == 'COMPLETED']
    task_list = task_list[['gpu_num', 'start_time', 'duration']]
    task_list = task_list[task_list['duration'] > 600]
    task_list = task_list[task_list['gpu_num'] > 0]
    task_list['start_time'] = pd.to_datetime(task_list['start_time'])
    task_list = task_list[task_list['start_time'] >= pd.to_datetime("2020-06-01 00:00:00")]
    task_list = task_list.reset_index(drop=True)
    task_list = task_list.loc[:num - 1, :]
    task_list['model_type'] = task_list.apply(lambda x: random.choice(LARGE_MODEL_LIST), axis=1)
    # print(task_list)
    task_list['npu_num'] = task_list.apply(get_npu_size, axis=1)
    task_list['task_iteration_num'] = 10
    task_list['TP'] = 8
    # pp_list = [1, 2, 4, 8, 16, 32]
    # task_list['PP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(pp_list)))
    task_list['PP'] = 1
    task_list['DP'] = task_list['npu_num'] // task_list.apply(lambda x: min(x['npu_num'], x['PP']), axis=1)
    # ep_list = [8, 16, 32] #[8, 16, 32]
    # task_list['EP'] = task_list['npu_num'].apply(lambda x: min(x, random.choice(ep_list)))
    task_list['EP'] = 16
    task_list['npu_num'] *= 8
    rng = np.random.default_rng(num)
    exponential_interval_list = rng.exponential(beta, num)
    task_list['arriving_time'] = np.cumsum(exponential_interval_list)
    new_task_info = task_list[['arriving_time', 'duration', 'model_type', 'npu_num', 'task_iteration_num',
                               'TP', 'PP', 'DP', 'EP']]
    new_task_info.loc[:, 'duration'] = new_task_info['duration'].apply(lambda x: int(3600 * random.uniform(0.5, 1.5)))
    print(new_task_info)
    new_task_info = new_task_info.values.tolist()
    print(len(new_task_info))
    return new_task_info


def randomly_chosen_accord_to_coff(class_map, random_seed):
    class_map_copy = copy.deepcopy(class_map)
    cur_sum = 0
    for key in class_map_copy:
        class_map_copy[key] = cur_sum + class_map_copy[key]
        cur_sum = class_map_copy[key]
    random.seed(random_seed)
    ran_pro = random.uniform(0, 1)
    chosen_res = -1
    for key in class_map_copy:
        if class_map_copy[key] > ran_pro:
            chosen_res = key
            break
    return chosen_res


def tmp_modify_occupied_NIC_num(task_occupied_NIC_num):
    if task_occupied_NIC_num <= 2:
        return 16
    elif task_occupied_NIC_num <= 4:
        return 32
    elif task_occupied_NIC_num <= 8:
        return 64
    else:
        return task_occupied_NIC_num


def random_exponential(lam):
    # x = np.arange(0, 15, 0.1)
    # y = lam * np.exp(-lam * x)
    pv = 0.0
    pv = (random.random() % 100) / 100
    while pv == 0:
        pv = (random.random() % 100) / 100

    pv = (-1 / lam) * log(1 - pv)
    print(pv)
    return pv


if __name__ == '__main__':
    print(get_fixed_requests_256_part_llm_tasks(100, 30000))
