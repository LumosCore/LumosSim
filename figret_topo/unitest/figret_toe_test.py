import sys
import numpy as np
from torch.utils.data import DataLoader


def test_figret_toe_env():
    from figret_topo.figret_helper import parse_args
    from figret_topo.src import FigretToEEnv
    props = parse_args(sys.argv[1:])
    env = FigretToEEnv(props)
    print(len(env.sorted_sd_pairs))
    sd_pairs = np.array(env.sorted_sd_pairs, dtype=int)
    sd_pairs = sd_pairs.reshape(16, 16, 15, 2)
    print(sd_pairs.shape)
    # for val in env.sorted_sd_pairs:
    #     print(val)


def test_toe_dataset_and_model():
    from figret_topo.figret_helper import parse_args
    from figret_topo.src import FigretDataset, FigretNetWork
    import torch
    props = parse_args(sys.argv[1:])
    train_dataset = FigretDataset(props.topo_name, props.hist_len, props.pod_num, props.spine_num_per_pod,
                                  props.train_hist_names, props.single_hist_size)
    train_dl = DataLoader(train_dataset, batch_size=props.batch_size, shuffle=True)
    while True:
        x, y = train_dl.__iter__().__next__()
        if torch.max(x) > 1e-5:
            break
    print(x.shape, y.shape)
    model = FigretNetWork(props.hist_len, props.pod_num, props.spine_num_per_pod, props.num_layer).double()
    y_pred = model(x)
    print(y_pred.shape)


def test_toe_model():
    from figret_topo.figret_helper import parse_args
    from figret_topo.src import FigretDataset, FigretNetWork
    props = parse_args(sys.argv[1:])
    train_dataset = FigretDataset(props.topo_name, props.hist_len, props.pod_num, props.spine_num_per_pod,
                                  props.train_hist_names, props.single_hist_size)
    train_dl = DataLoader(train_dataset, batch_size=props.batch_size, shuffle=True)
    x, y = train_dl.__iter__().__next__()
    model = FigretNetWork(props.hist_len, props.pod_num, props.spine_num_per_pod, props.num_layer).double()
    y_pred = model(x)
    print(y_pred.shape)


if __name__ == '__main__':
    test_toe_dataset_and_model()
