import os
import json
import sys
from networkx.readwrite import json_graph
from .linear_simulator import LinearSimulator
from .utils import paths_from_file
from figret.src.config import DATA_DIR


class LinearEnv:

    def __init__(self, props):
        self.topo_name = props.topo_name
        self.props = props
        self.init_topo()
        self.simulator = LinearSimulator(props)

    def init_topo(self):
        self.G = self.read_graph_json(self.topo_name)
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        self.pij = paths_from_file(paths_file="%s/%s/%s" % (DATA_DIR, self.topo_name, self.props.paths_file),
                                   num_nodes=self.num_nodes)

    def set_mode(self, mode):
        self.simulator.set_mode(mode)

    @staticmethod
    def read_graph_json(topo):
        with open(os.path.join(DATA_DIR, topo, topo + '.json'), 'r') as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)
