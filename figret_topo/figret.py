import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
from figret_helper import parse_args
from src import FigretToEEnv, FigretNetWork, FigretToE, FigretDataset, MODEL_DIR


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(521000)


def benchmark(props):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FigretToEEnv(props)
    figret = FigretToE(props, env, device)
    hist_len = props.hist_len
    pod_num = env.pod_num
    spine_num = env.spine_num_per_pod

    train_dataset = FigretDataset(props.topo_name, hist_len, pod_num, spine_num,
                                  props.train_hist_names, props.single_hist_size)
    train_dl = DataLoader(train_dataset, batch_size=props.batch_size, shuffle=True)

    eval_dataset = FigretDataset(props.topo_name, hist_len, pod_num, spine_num,
                                 props.test_hist_names, props.single_hist_size)
    eval_dl = DataLoader(eval_dataset, batch_size=props.batch_size, shuffle=False)

    model = FigretNetWork(props.hist_len, props.pod_num, props.spine_num_per_pod, props.num_layer).double()
    figret.set_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)
    figret.train(train_dl, eval_dl, optimizer)


if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)
