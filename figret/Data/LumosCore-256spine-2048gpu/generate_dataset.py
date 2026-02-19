from figret.benchmarks.linear.linear_src.utils import Get_edge_to_path
from figret.figret_helper import parse_args
# from rapidAIsim.scheduler.ocsexpander import TE_solver_lp, mesh_solver
from figret.benchmarks.linear.linear_src.linear_window_algorithm import LumosCore
from figret.benchmarks.linear.linear_src.linear_env import LinearEnv
import numpy as np
import json
from collections import defaultdict
import sys
import os
from concurrent.futures import ProcessPoolExecutor, wait


def init_tunnel_and_graph_json_file(pod_num, spine_num_per_pod, spine_up_port_num, single_link_capacity):
    # c_ij = TE_solver_lp.solve(pod_num, spine_num_per_pod, spine_up_port_num,
    #                           np.ones((pod_num, pod_num)) - np.eye(pod_num))
    # x_ijkt = mesh_solver.solve(spine_num_per_pod, spine_up_port_num, c_ij)
    x_ijkt = np.load('x_ijkt.npy')
    all_spine_num = pod_num * spine_num_per_pod
    spine_start_index = pod_num
    graph = dict()
    graph['directed'] = True
    graph["multigraph"] = False
    graph["graph"] = {}
    graph["nodes"] = [{"id": i} for i in range(pod_num + all_spine_num)]
    graph["links"] = []
    # 添加从pod到spine的link
    for i in range(pod_num):
        for t in range(spine_num_per_pod):
            spine_index = spine_start_index + i * spine_num_per_pod + t
            graph["links"].append({"source": i, "target": spine_index, "capacity": single_link_capacity * spine_up_port_num})
            graph["links"].append({"source": spine_index, "target": i, "capacity": single_link_capacity * spine_up_port_num})
    tunnels_direct = defaultdict(list)
    tunnels_multi_hop = defaultdict(list)
    # 添加直连tunnel和spine到spine的link
    for i in range(pod_num):
        for j in range(pod_num):
            if i == j:
                continue
            for t in range(spine_num_per_pod):
                link_num = np.sum(x_ijkt[i, j, :, t])
                if link_num == 0:
                    continue
                spine1_index = spine_start_index + i * spine_num_per_pod + t
                spine2_index = spine_start_index + j * spine_num_per_pod + t
                tunnels_direct[(i, j)].append((i, spine1_index, spine2_index, j))
                graph["links"].append({"source": spine1_index, "target": spine2_index, 
                                       "capacity": single_link_capacity * link_num})
    # 添加多跳tunnel
    for paths in tunnels_direct.values():
        for path in paths:
            i, spine1_index, spine2_index, j = path
            for k in range(pod_num):
                if k == i or k == j:
                    continue
                for path2 in tunnels_direct[(j, k)]:
                    if path2[1] != spine2_index:
                        continue
                    spine3_index = path2[2]
                    tunnels_multi_hop[(i, k)].append((i, spine1_index, spine2_index, spine3_index, k))
    with open('LumosCore-256spine-2048gpu.json', mode='w') as f:
        f.write(json.dumps(graph, indent=4))
    with open('tunnels.txt', mode='w') as f:
        for key, value in tunnels_direct.items():
            paths = []
            paths.extend(value)
            paths.extend(tunnels_multi_hop[key])
            f.write(f'{key[0]} {key[1]}:')
            f.write(",".join(['-'.join(map(str, path)) for path in paths]))
            f.write('\n')


def transform_log_to_hist(traffic_matrix_file, spine_switch_num, hist_name):
    with open(traffic_matrix_file) as f:
        demands = []
        for line in f:
            demand = line.split(',')[1]
            demand = map(float, demand.split(' '))
            demands.append(list(demand))
    demands = np.array(demands)
    pod_num = int(np.sqrt(demands.shape[1]))
    demands = demands.reshape((demands.shape[0], pod_num, pod_num))
    hist = np.zeros((demands.shape[0], pod_num + spine_switch_num, pod_num + spine_switch_num))
    hist[:, :pod_num, :pod_num] = demands
    hist = hist.reshape((hist.shape[0], hist.shape[1] * hist.shape[2]))
    # 去除头部和尾部的200个数据
    print(hist.shape)
    hist = hist[200:, :]
    hist = hist[:-200, :]
    hist_len = hist.shape[0]
    train_len = int(hist_len * 0.8)
    with open(f'train/{hist_name}.hist', mode='w') as f:
        for line in hist[:train_len]:
            f.write(" ".join(map(str, line)))
            f.write("\n")
    with open(f'test/{hist_name}.hist', mode='w') as f:
        for line in hist[train_len:]:
            f.write(" ".join(map(str, line)))
            f.write("\n")
    print("Train set length: {}, test set length: {}".format(
        train_len, hist_len - train_len))


def transform_log_to_hist_stage(traffic_file, spine_switch_num, start_hist_file, single_hist_len,
                                start_line_num, max_end_line_num, dir, offset):
    if start_line_num >= max_end_line_num:
        return
    read_file_ptr = open(traffic_file, mode='r')
    read_file_ptr.seek(offset, 0)
    for _ in range(start_line_num):
        next(read_file_ptr)
    
    demands = []
    for _ in range(min(single_hist_len, max_end_line_num - start_line_num)):
        line = read_file_ptr.readline()
        demand = line.split(',')[1]
        demand = map(float, demand.split(' '))
        demands.append(list(demand))
    demands = np.array(demands)
    pod_num = int(np.sqrt(demands.shape[1]))
    demands = demands.reshape((demands.shape[0], pod_num, pod_num))
    hist = np.zeros((demands.shape[0], pod_num + spine_switch_num, pod_num + spine_switch_num))
    hist[:, :pod_num, :pod_num] = demands
    hist = hist.reshape(hist.shape[0], hist.shape[1] * hist.shape[2])
    shape = np.array(hist.shape)
    # with open(f'{dir}/{start_hist_file}.hist', mode='w') as f:
    #     for line in hist:
    #         f.write(" ".join(map(str, line)))
    #         f.write("\n")
    np.savez_compressed(f'{dir}/{start_hist_file}.npz', arr_0=hist, shape=shape)


def transform_log_to_hist_multiprocess(traffic_matrix_file: str, spine_switch_num: int, file_length: int,
                                       start_hist_file: int, num_processers: int, single_hist_len: int,
                                       skip_len: int):
    """
    多线程地将traffic_matrix_file转换为hist文件。适用于traffic_matrix_file较大的情况。

    Args:
        traffic_matrix_file (str): 流量矩阵文件路径
        spine_switch_num (int): spine交换机总数量
        file_length (int): 流量矩阵文件的行数
        start_hist_file (int): 开始hist文件的编号
        num_processers (int): 并行度
        single_hist_len (int): 保存的单个hist文件的长度
        skip_len (int): 跳过的首尾行数
    """
    read_file_ptr = open(traffic_matrix_file, mode='r')
    # 跳过首尾
    for _ in range(skip_len):
        read_file_ptr.readline()
    file_length -= 2 * skip_len
    offset = read_file_ptr.tell()
    # 区分train和test
    train_length = int(file_length * 0.833)
    # train_length = 0
    stage_length = num_processers * single_hist_len
    stage_num = 0
    while stage_num * stage_length < train_length:
        params = [(traffic_matrix_file, spine_switch_num, start_hist_file + i, single_hist_len,
                   stage_num * stage_length + i * single_hist_len, train_length, 'train', offset)
                  for i in range(num_processers)]
        with ProcessPoolExecutor(num_processers) as executor:
            futures = [executor.submit(transform_log_to_hist_stage, *param) for param in params]
            wait(futures)
        stage_num += 1
        start_hist_file += num_processers
    # 计算test开始时文件的偏移量
    for _ in range(train_length):
        read_file_ptr.readline()
    offset = read_file_ptr.tell()
    file_length -= train_length
    start_hist_file -= num_processers * stage_num
    stage_num = 0
    while stage_num * stage_length < file_length:
        params = [(traffic_matrix_file, spine_switch_num, start_hist_file + i, single_hist_len,
                   stage_num * stage_length + i * single_hist_len, file_length, 'test', offset)
                  for i in range(num_processers)]
        with ProcessPoolExecutor(num_processers) as executor:
            futures = [executor.submit(transform_log_to_hist_stage, *param) for param in params]
            wait(futures)
        stage_num += 1
        start_hist_file += num_processers


def generate_demand(props):
    env = LinearEnv(props)
    edge_to_path = Get_edge_to_path(env.G, env.pij)
    lumos_core = LumosCore(props, env.G, env.pij, edge_to_path)

    if props.train_hist_names != '':
        mlu_list = []
        for hist in env.simulator.train_hist.tms:
            mlu, _ = lumos_core.solve_traffic_engineering(hist)
            print(mlu)
            mlu_list.append(mlu)
        with open(f'train/{props.train_hist_names}.opt', mode='w') as f:
            f.write("\n".join(map(str, mlu_list)))

    if props.test_hist_names != '':
        mlu_list = []
        for hist in env.simulator.test_hist.tms:
            mlu, _ = lumos_core.solve_traffic_engineering(hist)
            print(mlu)
            mlu_list.append(mlu)
        with open(f'test/{props.test_hist_names}.opt', mode='w') as f:
            f.write("\n".join(map(str, mlu_list)))


def check_demand(traffic_log, topo_file, capacity, pod_num):
    x_ijkt = np.load(topo_file)
    x_ij = np.sum(x_ijkt, axis=(2, 3))
    with open(traffic_log) as f:
        for line in f:
            demand = line.split(',')[1]
            demand = map(float, demand.split(' '))
            demand = np.array(list(demand)).reshape((pod_num, pod_num))
            for i in range(pod_num):
                for j in range(pod_num):
                    if i == j:
                        continue
                    if demand[i, j] > 5000 * x_ij[i, j] * capacity:
                        print(f'Error: demand from {i} to {j} is {demand[i, j]}, capacity is {x_ij[i, j] * capacity}')


def add_shape_info_to_npz_file():
    files = os.listdir('train')
    for file in files:
        if file in ['48.npz', '49.npz', '50.npz']:
            data = np.load(f'train/{file}')
            hist = data['arr_0']
            shape = np.array(hist.shape)
            np.savez_compressed(f'train/{file}', arr_0=hist, shape=shape)


def read_opt():
    opts = []
    with open('train/1.opt') as f:
        for line in f:
            opts.append(float(line))
    opts.sort(reverse=True)
    print(opts)


if __name__ == '__main__':
    # check_demand('traffic_matrix.log', 'x_ijkt.npy', 1600, 32)
    # read_opt()
    # init_tunnel_and_graph_json_file(32, 8, 8, 1600)
    props = parse_args(sys.argv[1:])
    generate_demand(props)
    # transform_log_to_hist_multiprocess(
    #     'traffic_matrix.log', 256, 4560, 3, 3, 2000, 300
    # )
    # add_shape_info_to_npz_file()
