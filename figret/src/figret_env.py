import os
import json
import numpy as np
from networkx.readwrite import json_graph
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict

from .figret_simulator import FigretSimulator
from .utils import normalize_size


class FigretEnv:

    def __init__(self, props):
        """Initialize the FigretEnv with the properties.

        Args:
            props: arguments from the command line
        """
        self.topo_name = props.topo_name
        self.props = props
        self.set_dirs(props)
        self.init_topo()
        self.simulator = FigretSimulator(props, self.num_nodes)

    def set_dirs(self, props):
        """根据props设置DATA_DIR，MODEL_DIR和RESULT_DIR"""
        if props.data_dir != 'default':
            if not os.path.exists(props.data_dir):
                raise FileNotFoundError('Data directory does not exist')
            exp_dir = os.path.join(props.data_dir, '..')
            from . import config
            config.DATA_DIR = props.data_dir
            config.MODEL_DIR = os.path.join(exp_dir, 'Model')
            config.RESULT_DIR = os.path.join(exp_dir, 'Result')
            config.init_dirs()

    def init_topo(self):
        """Initialize the topology information with the given name."""
        self.G = self.read_graph_json(self.topo_name)
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        self.adj = self.get_adj()
        self.pij = self.paths_from_file()
        self.edges_map, self.capacity = self.get_edges_map()
        self.paths_to_edges = self.get_paths_to_edges(self.pij)
        self.num_paths = self.paths_to_edges.shape[0]
        self.commodities_to_paths = self.get_commodities_to_paths()
        self.commodities_to_path_nums = self.get_commodities_to_path_nums()
        self.constant_pathlen = self.is_path_length_constant(self.commodities_to_path_nums)

    def set_mode(self, mode):
        """Set the mode of the simulator.

        Args:
            mode: train or test
        """
        self.simulator.set_mode(mode)

    def read_graph_json(self, topo_name):
        """Read the graph from the json file.

        Args:
            topo_name: name of the topology
        """
        from .config import DATA_DIR
        print(DATA_DIR)
        with open(os.path.join(DATA_DIR, topo_name, topo_name + '.json'), 'r') as f:
            data = json.load(f)
        g = json_graph.node_link_graph(data)
        return g

    def paths_from_file(self):
        """Read the candidate paths from the file."""
        from .config import DATA_DIR
        paths_file = "%s/%s/%s" % (DATA_DIR, self.topo_name, self.props.paths_file)
        pij = defaultdict(list)
        with open(paths_file, 'r') as f:
            lines = sorted(f.readlines())
            for line in lines:
                src, dst = line.split(':')[0].split(' ')
                paths = line.split(':')[1].split(',')
                for p_ in paths:
                    node_list = list(map(int, p_.split('-')))
                    pij[(int(src), int(dst))].append(self.node_to_path(node_list))
        return pij

    def node_to_path(self, node_list):
        """Convert the node list path to the edge list path."""
        return [(v1, v2) for v1, v2 in zip(node_list, node_list[1:])]

    def get_edges_map(self):
        """Get the map from the edge to the edge id."""
        eid = 0
        edges_map = dict()
        capacity = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj[i, j] == 1:
                    edges_map[(i, j)] = eid
                    capacity.append(normalize_size(self.G[i][j]['capacity']))
                    eid += 1
        return edges_map, capacity

    def get_adj(self):
        """Get the adjacency matrix of the graph."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d:
                    continue
                if d in self.G[s]:
                    adj[s, d] = 1
        return adj

    def get_paths_to_edges(self, paths):
        """Get the paths_to_edges matirx, [num_paths, num_edges]
           paths_to_edges[i, j] = 1 if edge j is in path i

        Args:
            paths: the candidate paths
        """
        paths_arr = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                for p in paths[(i, j)]:
                    path_edges_id = [self.edges_map[e] for e in p]
                    path_onehot_vector = np.zeros((int(self.num_edges),))
                    for eid in path_edges_id:
                        path_onehot_vector[eid] = 1
                    paths_arr.append(path_onehot_vector)
        return csr_matrix(np.stack(paths_arr))

    def get_commodities_to_paths(self):
        """Get the commodities_to_paths matrix, [num_commodities, num_paths]
           commodities_to_paths[i, j] = 1 if path j is a candidate path for commodity i
        """
        commodities_to_paths = lil_matrix((self.num_nodes * (self.num_nodes - 1), self.num_paths))
        commid = 0
        pathid = 0
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                for _ in self.pij[(src, dst)]:
                    commodities_to_paths[commid, pathid] = 1
                    pathid += 1
                commid += 1
        return csr_matrix(commodities_to_paths)

    def get_commodities_to_path_nums(self):
        """Get the number of candidate paths for each commodity."""
        path_num_per_commdities = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                path_num_per_commdities.append(len(self.pij[(src, dst)]))
        return path_num_per_commdities

    def is_path_length_constant(self, lst):
        """Check if all path len in the list are the same."""
        assert len(lst) > 0
        return lst.count(lst[0]) == len(lst)
