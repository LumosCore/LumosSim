import os
import sys
from typing import List
import numpy as np
import networkx as nx
from tqdm import tqdm

sys.path.append(os.path.join('..', '..'))
from src.config import RESULT_DIR, DATA_DIR
from src.utils import print_to_txt

from linear_src.linear_env import LinearEnv
from linear_src.utils import Get_edge_to_path, paths_from_file
from linear_src.linear_routing import Routing

from linear_helper import parse_args
from tools.matrix_to_topo import k_shortest_simple_paths


def _read_topos_from_topo_txt(topo_txt_path: str, N: int) -> List[np.ndarray]:
    """从 topo.txt 读取每一时刻的邻接矩阵（整数权，0/1/多链路数）。

    每行是按行优先展开的 N*N 数列，逗号或空格分隔。
    返回长度为 T 的列表，每个元素是形状 (N,N) 的 np.ndarray[int]。
    """
    topos: List[np.ndarray] = []
    with open(topo_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            vals = [int(float(x)) for x in parts]
            if len(vals) != N * N:
                raise ValueError(f'topo.txt row length {len(vals)} != N*N {N*N}')
            mat = np.array(vals, dtype=int).reshape((N, N))
            # 对角线清零（不考虑自环）
            for i in range(N):
                mat[i, i] = 0
            topos.append(mat)
    return topos


def _build_graph_from_adj(adj: np.ndarray, base_capacity: float = 10000.0) -> nx.DiGraph:
    """将邻接矩阵转为有向图，容量=base_capacity*权值。"""
    n = adj.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = int(adj[i, j])
            if w != 0:
                G.add_edge(i, j, capacity=float(base_capacity) * float(w))
    return G


def benchmark(props):
    # 先用固定拓扑环境获取 N、TM 列表与 paths 文件名
    env = LinearEnv(props)
    tm_list = env.simulator.test_hist.tms
    N = env.G.number_of_nodes()

    # 从 Data/<topo_name>/topo.txt 读取每时刻拓扑
    topo_txt_path = os.path.join(DATA_DIR, props.topo_name, 'topo.txt')
    if not os.path.exists(topo_txt_path):
        raise FileNotFoundError(f'Missing topo.txt: {topo_txt_path}')
    topos = _read_topos_from_topo_txt(topo_txt_path, N)

    if len(topos) != len(tm_list):
        # 若 topo 行数与 TM 数不同，按较短的对齐
        min_len = min(len(topos), len(tm_list))
        tm_list = tm_list[:min_len]
        topos = topos[:min_len]

    # 为每个时刻构建对应拓扑与候选路径，并计算 MLU
    mlu_list = []
    for adj, tm in tqdm(zip(topos, tm_list), total=len(topos), desc='solve optimal mlu per TM with its topo'):
        G_t = _build_graph_from_adj(adj)
        # 按当前拓扑直接重新计算 pij：每个 (s,d) 的 k 条最短路
        pij = {}
        k = getattr(props, 'path_num', 3)
        for s in range(N):
            for d in range(N):
                if s == d:
                    continue
                paths = k_shortest_simple_paths(G_t, s, d, k)
                pij[(s, d)] = paths

        # 更新 env 中的拓扑与路径，供其他组件使用
        env.update_topo(G_t, pij)

        edge_to_path = Get_edge_to_path(env.G, env.pij)
        algorithm = Routing(env.G, env.pij, edge_to_path)

        res = algorithm.MLU_traffic_engineering([tm])
        # import pdb; pdb.set_trace()
        if not res:
            pass
        else:
            mlu, _ = res
            mlu_list.append(mlu)

    result_save_path = os.path.join(RESULT_DIR,'Facebook', props.topo_name, 'result.txt')
    print_to_txt(mlu_list, result_save_path)


if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)


