import os
import numpy as np
from tqdm import tqdm
from .utils import linear_get_dir, linear_get_hists_from_folder


class LinearSimulator:
    def __init__(self, props):
        self.props = props
        self.set_dirs(props)
        self.get_train_test_file(props)

    def get_train_test_file(self, props):
        data_train_folder = linear_get_dir(props, is_test=False)
        data_test_folder = linear_get_dir(props, is_test=True)
        if props.train_hist_names != 'all':
            train_hist_files = []
            if props.train_hist_names != '':
                hist_names = self.deal_with_hist_names(props.train_hist_names)
                for hist_name in hist_names:
                    train_hist_files.append(f'{data_train_folder}/{hist_name}.hist')
        else:
            train_hist_files = linear_get_hists_from_folder(data_train_folder)

        if props.test_hist_names != 'all':
            test_hist_files = []
            if props.test_hist_names != '':
                hist_names = self.deal_with_hist_names(props.test_hist_names)
                for hist_name in hist_names:
                    test_hist_files.append(f'{data_test_folder}/{hist_name}.hist')
        else:
            test_hist_files = linear_get_hists_from_folder(data_test_folder)

        self.train_hist = Histories(train_hist_files, 'train')
        self.test_hist = Histories(test_hist_files, 'test')

        # 以下仅用于对Jupiter的测试
        # self.test_hist.opts = self.test_hist.opts[:self.props.hist_len]
        # train_tms = self.train_hist.tms
        # self.train_hist.tms = []
        # for i in range(self.props.hist_len, len(train_tms)):
        #     hist = train_tms[i - self.props.hist_len:i]
        #     self.train_hist.tms.append(hist)

    def set_dirs(self, props):
        """根据props设置DATA_DIR，MODEL_DIR和RESULT_DIR"""
        if props.data_dir != 'default':
            if not os.path.exists(props.data_dir):
                raise FileNotFoundError('Data directory does not exist')
            exp_dir = os.path.join(props.data_dir, '..')
            from figret.src import config
            config.DATA_DIR = props.data_dir
            config.MODEL_DIR = os.path.join(exp_dir, 'Model')
            config.RESULT_DIR = os.path.join(exp_dir, 'Result')
            config.init_dirs()
    
    def set_mode(self, mode):
        hist_str = 'self.' + mode + '_hist'
        self.cur_hist = eval(hist_str)

    @staticmethod
    def deal_with_hist_names(hist_names):
        if isinstance(hist_names, str):
            hist_names = list(hist_names.split(','))
        res = []
        for hist_name in hist_names:
            if '-' in hist_name:
                start, end = map(int, hist_name.split('-'))
                res.extend(range(start, end + 1))
            else:
                res.append(int(hist_name))
        return res


class Histories:

    def __init__(self, tm_files=None, htype=None):
        self.tms = []
        self.opts = []
        self.htype = htype

        for fname in tm_files:
            print('[+] Populating Tms for file: {}'.format(fname))
            self.populate_tms(fname)
            self.read_opts(fname)
        if self.tms:
            self.tms = np.vstack(self.tms)

    def read_opts(self, fname):
        try:
            with open(fname.replace('hist', 'opt')) as f:
                lines = f.readlines()
                self.opts += [np.float64(_) for _ in lines]
        except:
            return None

    def populate_tms(self, fname):
        hist_name = fname.split('/')[-1].split('.')[0]
        npy_file = f'{"/".join(fname.split("/")[:-1])}/{hist_name}.npy'
        npz_file = f'{"/".join(fname.split("/")[:-1])}/{hist_name}.npz'
        if os.path.exists(npz_file):
            tm = np.load(npz_file)
            tm = tm['arr_0'].squeeze()
            self.tms.append(tm)
        elif os.path.exists(npy_file):
            tm = np.load(npy_file).squeeze()
            self.tms.append(tm)
        else:
            self.tms.append([])
            with open(fname, 'r') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                tm = self.parse_tm_line(line)
                self.tms[-1].append(tm)
            self.tms[-1] = np.array(self.tms[-1])
            np.savez_compressed(npz_file, self.tms[-1])

    def parse_tm_line(self, line):
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype=np.float64)
        return tm
