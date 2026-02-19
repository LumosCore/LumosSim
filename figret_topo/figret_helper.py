import argparse


def add_default_args(parser):
    parser.add_argument('--topo_name', type=str, default='Facebook_tor_a')
    parser.add_argument('--train_hist_names', type=str, default='',
                        help="'all' means all files in the directory, "
                             "'1,2,3' means 1.hist, 2.hist and 3.hist, "
                             "'4-6' means 4.hist, 5.hist and 6.hist, "
                             "comma and dash can be combined, "
                             "other formats are not supported")
    parser.add_argument('--test_hist_names', type=str, default='',
                        help="see train_hist_names")
    parser.add_argument('--data_dir', type=str, default='default',
                        help="directory of the data files. If is default, "
                             "the default directory is the current file's "
                             "directory joined with 'DATA'")
    parser.add_argument('--single_hist_size', type=int, default=2000)
    parser.add_argument('--dataset_label', type=str, default='',
                        help="the label is used to distinguish different "
                             "datasets when saving models, format is "
                             "\{beta_value\}_\{statistic_interval\}")
    parser.add_argument('--spine_num_per_pod', type=int, default=8)
    parser.add_argument('--pod_num', type=int, default=16)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hist_len', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=3.)
    parser.add_argument('--direct_ratio', type=float, default=0.9)
    parser.add_argument('--topk_ratio', type=float, default=1)

    return parser


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)

    return parser.parse_args(args)
