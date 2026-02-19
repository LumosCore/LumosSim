import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict


class FigretToEEnv:

    def __init__(self, props):
        """Initialize the FigretToEEnv with the properties.

        Args:
            props: arguments from the command line
        """
        self.topo_name = props.topo_name
        self.props = props
        self.set_dirs(props)

        # Initialize the topology information
        self.spine_num_per_pod = self.props.spine_num_per_pod
        self.pod_num = self.props.pod_num
        self.sd_pairs, self.sorted_sd_pairs = self.generate_sd_pairs()
        self.pij = self.generate_paths()
        self.edges_map, self.num_edges = self.get_edges_map()
        self.paths_to_edges = self.get_paths_to_edges(self.pij)
        self.num_paths = self.paths_to_edges.shape[0]
        self.commodities_to_paths, self.commodities_to_path_nums = self.get_commodities_to_paths_and_nums()
        self.constant_pathlen = self.is_path_length_constant(self.commodities_to_path_nums)

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

    def generate_sd_pairs(self):
        """Generate all source-destination pairs."""
        sd_pairs = set()
        for i in range(self.pod_num):
            for j in range(self.spine_num_per_pod):
                for k in range(self.pod_num):
                    if i == k:
                        continue
                    src = i * self.spine_num_per_pod + j
                    dst = k * self.spine_num_per_pod + j
                    sd_pairs.add((src, dst))
        return sd_pairs, list(sorted(sd_pairs))

    def generate_paths(self):
        """生成所有节点对之间的路径"""
        pij = defaultdict(list)
        for src, dst in self.sorted_sd_pairs:
            paths = [[(src, dst)]]
            i = src // self.spine_num_per_pod
            j = src % self.spine_num_per_pod
            k = dst // self.spine_num_per_pod
            for l in range(self.pod_num):
                if l == i or l == k:
                    continue
                mid = l * self.spine_num_per_pod + j
                paths.append([(src, mid), (mid, dst)])
            pij[(src, dst)] = paths
        return pij

    def get_edges_map(self):
        """Get the map from the edge to the edge id."""
        eid = 0
        edges_map = dict()
        for i, j in self.sorted_sd_pairs:
            edges_map[(i, j)] = eid
            eid += 1
        return edges_map, eid

    def get_paths_to_edges(self, paths):
        """Get the paths_to_edges matirx, [num_paths, num_edges]
           paths_to_edges[i, j] = 1 if edge j is in path i

        Args:
            paths: the candidate paths
        """
        paths_arr = []
        for i, j in self.sorted_sd_pairs:
            for p in paths[(i, j)]:
                path_edges_id = [self.edges_map[e] for e in p]
                path_onehot_vector = np.zeros((self.num_edges,))
                for eid in path_edges_id:
                    path_onehot_vector[eid] = 1
                paths_arr.append(path_onehot_vector)
        return csr_matrix(np.stack(paths_arr))

    def get_commodities_to_paths_and_nums(self):
        """
        Get the commodities_to_paths matrix, [num_commodities, num_paths] and each commodity's path number
        commodities_to_paths[i, j] = 1 if path j is a candidate path for commodity i
        """
        commodities_to_paths = lil_matrix((len(self.sd_pairs), self.num_paths))
        path_num_per_commdities = []
        commid = 0
        pathid = 0
        for src, dst in sorted(self.sd_pairs):
            for _ in self.pij[(src, dst)]:
                commodities_to_paths[commid, pathid] = 1
                pathid += 1
            commid += 1
            path_num_per_commdities.append(len(self.pij[(src, dst)]))
        return csr_matrix(commodities_to_paths), path_num_per_commdities

    def is_path_length_constant(self, lst):
        """Check if all path len in the list are the same."""
        assert len(lst) > 0
        return lst.count(lst[0]) == len(lst)
