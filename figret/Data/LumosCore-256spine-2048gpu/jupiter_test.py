from figret.benchmarks.linear.linear_src.utils import Get_edge_to_path
from figret.figret_helper import parse_args
from figret.benchmarks.linear.linear_src.linear_window_algorithm import Jupiter
from figret.benchmarks.linear.linear_src.linear_env import LinearEnv
import sys
import torch
import numpy as np

from figret.src import FigretEnv


def generate_demand(props):
    env = LinearEnv(props)
    edge_to_path = Get_edge_to_path(env.G, env.pij)
    jupiter = Jupiter(props, env.G, env.pij, edge_to_path)

    figret_env = FigretEnv(props)
    ctp_coo = figret_env.commodities_to_paths.tocoo()
    commodities_to_paths = torch.sparse_coo_tensor(
        np.vstack((ctp_coo.row, ctp_coo.col)),
        torch.DoubleTensor(ctp_coo.data),
        torch.Size(ctp_coo.shape))  # shape: (num_commodities, num_paths)
    pte_coo = figret_env.paths_to_edges.tocoo()
    paths_to_edges = torch.sparse_coo_tensor(
        np.vstack((pte_coo.row, pte_coo.col)),
        torch.DoubleTensor(pte_coo.data),
        torch.Size(pte_coo.shape))  # shape: (num_paths, num_edges)
    edges_capacity = torch.tensor(figret_env.capacity).unsqueeze(1)

    tms = env.simulator.test_hist.tms
    opts = env.simulator.test_hist.opts

    loss_list = []
    with torch.no_grad():
        for i in range(props.hist_len, len(tms)):
            hist = tms[i - props.hist_len:i]
            y_true = tms[i]
            mlu, split_ratios = jupiter.solve_traffic_engineering(hist)
            tmp_demand_on_paths = commodities_to_paths.transpose(0, 1).matmul(
                y_true.transpose(0, 1))  # shape: (num_paths, 1)
            demand_on_paths = tmp_demand_on_paths.mul(split_ratios)  # shape: (num_paths, 1)
            flow_on_edges = paths_to_edges.transpose(0, 1).matmul(demand_on_paths)  # shape: (num_edges, 1)
            congestion = flow_on_edges.divide(edges_capacity)  # shape: (num_edges, 1)
            mean_cong = torch.mean(congestion.flatten(), dim=0)
            max_cong = torch.max(congestion.flatten(), dim=0).values
            loss = (mean_cong + max_cong).item() / opts[i]
            loss_list.append(loss)

    with open('test/jupiter.opt', mode='w') as f:
        f.write("\n".join(map(str, loss_list)))


def generate_demand2(props):
    env = LinearEnv(props)
    edge_to_path = Get_edge_to_path(env.G, env.pij)
    jupiter = Jupiter(props, env.G, env.pij, edge_to_path)

    tms = env.simulator.test_hist.tms
    opts = env.simulator.test_hist.opts

    loss_list = []
    for i in range(props.hist_len, len(tms)):
        hist = tms[i - props.hist_len:i]
        mlu, split_ratios = jupiter.solve_traffic_engineering(hist)
        if opts[i - 1] == 0:
            loss = 1.0
        else:
            loss = mlu / opts[i - 1]
        print(loss)
        loss_list.append(loss)

    with open('test/jupiter.opt', mode='w') as f:
        f.write("\n".join(map(str, loss_list)))
    print(np.mean(loss_list))


if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    generate_demand2(props)
