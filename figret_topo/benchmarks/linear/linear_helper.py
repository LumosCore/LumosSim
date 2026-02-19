import argparse


def add_default_args(parser):

    parser.add_argument('--topo_name', type=str, default='Facebook_tor_a')
    parser.add_argument('--paths_file', type=str, default='tunnels.txt')
    parser.add_argument('--path_num', type=int, default=3)
    parser.add_argument('--train_hist_names', type=str, default='',
                        help="'all' means all files in the directory, "
                             "'1,2,3' means 1.hist, 2.hist and 3.hist, "
                             "other formats are not supported")
    parser.add_argument('--test_hist_names', type=str, default='',
                        help="see train_hist_names")
    parser.add_argument('--data_dir', type=str, default='default',
                        help="directory of the data files. If is default, "
                             "the default directory is the current file's "
                             "directory joined with 'DATA'")

    parser.add_argument('--hist_len', type=int, default=12)

    parser.add_argument('--TE_solver', type=str, default='Jupiter')

    parser.add_argument('--beta', type=float, default=1, help='the weight of the ALU')
    parser.add_argument('--gamma', type=float, default=3, help='the weight of the ToE')

    # Jupiter
    parser.add_argument('--spread', type=float, default=0.5)

    # COPE
    parser.add_argument('--budget', action='store_true', default=False)

    return parser


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)

    return parser.parse_args(args)
