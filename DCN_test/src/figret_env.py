import os
import json
import numpy as np
import torch
import pdb

from networkx.readwrite import json_graph
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict

from .config import DATA_DIR
from .figret_simulator import FigretSimulator
from .utils import normalize_size

class FigretEnv():

    def __init__(self, props):
        """用给定参数初始化环境"""
        self.topo_name = props.topo_name  # 拓扑名称
        self.props = props  # 参数
        self.init_topo()  # 初始化拓扑结构
        self.simulator = FigretSimulator(props, self.num_nodes)  # 初始化流量仿真器

    def init_topo(self):
        """初始化拓扑相关信息"""
        self.G = self.read_graph_json(self.topo_name)  # 读取拓扑图
        self.num_nodes = self.G.number_of_nodes()  # 节点数
        self.num_edges = self.G.number_of_edges()  # 边数
        self.adj = self.get_adj()  # 邻接矩阵
        self.pij = self.paths_from_file()  # 处理路径到边的映射，存在一个字典中
        self.edges_map, self.capacity = self.get_edges_map()  # 边到编号的映射和容量
        self.paths_to_edges = self.get_paths_to_edges(self.pij)  # 路径到边的稀疏矩阵
        self.num_paths = self.paths_to_edges.shape[0]  # 路径总数
        self.commodities_to_paths = self.get_commodities_to_paths()  # 业务到路径的稀疏矩阵
        self.commodities_to_path_nums = self.get_commodities_to_path_nums()  # 每个业务的路径数
        self.constant_pathlen = self.is_path_length_constant(self.commodities_to_path_nums)  # 路径数是否一致

    def set_mode(self, mode):
        """设置仿真器的模式（训练或测试）"""
        self.simulator.set_mode(mode)

    def read_graph_json(self, topo_name):
        """从json文件读取拓扑图"""
        with open(os.path.join(DATA_DIR, topo_name, topo_name + '.json'), 'r') as f:
            data = json.load(f)
        g = json_graph.node_link_graph(data)
        return g
    
    def paths_from_file(self):
        """从文件读取所有候选路径"""
        paths_file = "%s/%s/%s"%(DATA_DIR, self.topo_name, self.props.paths_file)
        pij = defaultdict(list)
        pid = 0
        with open(paths_file, 'r') as f:
            lines = sorted(f.readlines())
            lines_dict = {line.split(":")[0] : line for line in lines if line.strip() != ""}
            for src in range(self.num_nodes):
                for dst in range(self.num_nodes):
                    if src == dst:
                        continue
                    try:
                        # 优先用字典查找，找不到再遍历
                        if "%d %d" % (src, dst) in lines_dict:
                            line = lines_dict["%d %d" % (src, dst)].strip()
                        else:
                            line = [l for l in lines if l.startswith("%d %d:" % (src, dst))]
                            if line == []:
                                continue
                            line = line[0]
                            line = line.strip()
                        if not line: continue
                        i,j = list(map(int, line.split(":")[0].split(" ")))
                        paths = line.split(":")[1].split(",")
                        for p_ in paths:
                            node_list = list(map( int, p_.split("-")))
                            pij[(i, j)].append(self.node_to_path(node_list))
                            pid += 1
                    except Exception as e:
                        print(e)
                        pdb.set_trace()
        # pdb.set_trace()
        return pij
    
    def node_to_path(self, node_list):
        """将节点序列转为边序列"""
        return [(v1, v2) for v1, v2 in zip(node_list, node_list[1:])]
    
    def get_edges_map(self):
        """获取边到编号的映射和容量列表"""
        eid = 0
        edges_map = dict()
        capacity = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj[i,j] == 1:
                    edges_map[(i,j)] = eid
                    capacity.append(normalize_size(self.G[i][j]['capacity']))
                    eid += 1
        # pdb.set_trace()
        return edges_map, capacity
    
    def get_adj(self):
        """获取邻接矩阵"""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d:
                    continue
                if d in self.G[s]:
                    adj[s,d] = 1
        return adj
    
    def get_paths_to_edges(self, paths):
        """生成路径到边的稀疏矩阵 [num_paths, num_edges]
           paths_to_edges[i, j] = 1 表示第i条路径包含第j条边
        Args:
            paths: 候选路径
        """
        paths_arr = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                for p in paths[(i, j)]:
                    p_ = [self.edges_map[e] for e in p]
                    p__ = np.zeros((int(self.num_edges),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
                    # pdb.set_trace()
        return csr_matrix(np.stack(paths_arr))
    
    def get_commodities_to_paths(self):
        """生成业务到路径的稀疏矩阵 [num_commodities, num_paths]
           commodities_to_paths[i, j] = 1 表示第j条路径属于第i个业务
        """
        commodities_to_paths = lil_matrix((self.num_nodes * (self.num_nodes - 1), self.num_paths))
        commid = 0
        pathid = 0
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                for _ in self.pij[(src,dst)]:
                    commodities_to_paths[commid, pathid] = 1
                    pathid += 1
                commid += 1
        # pdb.set_trace()
        return csr_matrix(commodities_to_paths)

    def get_commodities_to_path_nums(self):
        """获取每个业务的候选路径数"""
        path_num_per_commdities = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                path_num_per_commdities.append(len(self.pij[(src, dst)]))
        # pdb.set_trace()
        return path_num_per_commdities
    
    def is_path_length_constant(self, lst):
        """判断所有业务的候选路径数是否一致"""
        assert len(lst) > 0
        return lst.count(lst[0]) == len(lst)
