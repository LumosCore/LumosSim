import numpy as np
import random

# 根据给定的端口数量和流数量生成冲突概率，然后根据概率生成端口占用实例

def calc_conflict_pro(n, m, test_total_time=100000):
    spine_conflict_state = {}
    for test_time in range(test_total_time):
        outport_list_used_num = []
        for i in range(n):
            outport_list_used_num.append(0)
        for leaf_port_index in range(m):
            spine_port_index_temp = random.randint(0, n-1)
            outport_list_used_num[spine_port_index_temp] = outport_list_used_num[spine_port_index_temp]+1
        # print(outport_list_used_num)
        for used_num in outport_list_used_num:
            if used_num not in spine_conflict_state:
                spine_conflict_state[used_num] = 0
            spine_conflict_state[used_num] = spine_conflict_state[used_num]+1
    port_list = []
    probability_list = []
    for key, value in spine_conflict_state.items():
        spine_conflict_state[key] = spine_conflict_state[key] / \
            (test_total_time*n)
        port_list.append(key)
        probability_list.append(value/(test_total_time*n))
    print(port_list)
    print(probability_list)
    ret = p_random(port_list, probability_list)
    print(ret)
    random.shuffle(ret)
    return ret


def p_random(arr1, arr2):
    assert len(arr1) == len(arr2), "Length does not match."
    assert abs(sum(arr2)-1) <= 0.0000001, "Total rate is not 1."

    data = np.random.choice(arr1, len(arr1), p=arr2)
    return data

# # 采样生成端口占用实例
# def calc_conflict(n, m):
#     spine_conflict_state = {}
#     outport_list_used_num = []
#     for i in range(n):
#         outport_list_used_num.append(0)
#     for leaf_port_index in range(m):
#         spine_port_index_temp = random.randint(0, n-1)
#         outport_list_used_num[spine_port_index_temp] = outport_list_used_num[spine_port_index_temp]+1
#     return outport_list_used_num

# 采样生成端口占用实例


def calc_conflict(n):
    inport_bandwidth_list = []
    for i in range(n):
        inport_bandwidth_list.append(1)
    inport_connect_list = {}
    outport_list_used_num = {}
    for leaf_port_index in range(n):
        spine_port_index_temp = random.randint(0, n - 1)
        if spine_port_index_temp not in outport_list_used_num:
            outport_list_used_num[spine_port_index_temp] = 0
        outport_list_used_num[spine_port_index_temp] = outport_list_used_num[spine_port_index_temp] + 1
        inport_connect_list[leaf_port_index] = spine_port_index_temp
    for i in range(len(inport_bandwidth_list)):
        inport_bandwidth_list[i] = inport_bandwidth_list[i] / \
            outport_list_used_num[inport_connect_list[i]]
    temp_total_sum = sum(inport_bandwidth_list)
    for i in range(len(inport_bandwidth_list)):
        inport_bandwidth_list[i] = inport_bandwidth_list[i] / temp_total_sum
    # print(inport_connect_list)
    # print(outport_list_used_num)
    return inport_bandwidth_list


if __name__ == "__main__":
    print(calc_conflict(1))
    print(calc_conflict(3))
