import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
from figret_helper import parse_args
from src import FigretEnv, FigretNetWork, Figret, FigretDataset, MODEL_DIR


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
    env = FigretEnv(props)
    figret = Figret(props, env, device)

    if props.mode == 'train':
        train_dataset = FigretDataset(props.topo_name, props.hist_len, env.num_nodes, props.train_hist_names,
                                      'train', props.single_hist_size)
        train_dl = DataLoader(train_dataset, batch_size=props.batch_size, shuffle=True)
        test_dataset = FigretDataset(props.topo_name, props.hist_len, env.num_nodes, props.test_hist_names,
                                     'test', props.single_hist_size)
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model = FigretNetWork(props.hist_len * env.num_nodes * (env.num_nodes - 1), env.num_paths,
                              props.num_layer).double()
        figret.set_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)
        figret.train(train_dl, test_dl, optimizer)
    elif props.mode == 'test':
        test_dataset = FigretDataset(props.topo_name, props.hist_len, env.num_nodes, props.test_hist_names,
                                     'test', props.single_hist_size)
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model_name = f'{MODEL_DIR}/{props.topo_name}'
        if not os.path.exists(f'{model_name}.pt'):
            model_name = f'{model_name}_{props.hist_len}_{props.num_layer}_{props.dataset_label}.pt'
        else:
            model_name = f'{model_name}.pt'
        print("Loading model from: ", os.path.basename(model_name))
        model = FigretNetWork(props.hist_len * env.num_nodes * (env.num_nodes - 1), env.num_paths,
                              props.num_layer).double()
        model.load_state_dict(torch.load(model_name))
        figret.set_model(model)
        figret.test(test_dl)


if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)
