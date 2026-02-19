from collections import defaultdict
from functools import lru_cache
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from typing import List, Union, Literal

from .config import RESULT_DIR, MODEL_DIR, DATA_DIR
from .utils import print_to_txt, normalize_size


class Figret:
    """Figret class for training and testing the model."""

    def __init__(self, props, env, device):
        """Initialize the Figret with the properties, environment, model and device.

        Args:
            props: arguments from the command line
            env: environment for the Figret
            device: GPU or CPU
            model: FigretNetWork model
        """
        self.props = props
        self.env = env
        self.device = device
        self.model = None
        self.optimizer = None
        self.dataset_label = props.dataset_label
        ctp_coo = env.commodities_to_paths.tocoo()
        self.commodities_to_paths = torch.sparse_coo_tensor(
            np.vstack((ctp_coo.row, ctp_coo.col)),
            torch.DoubleTensor(ctp_coo.data),
            torch.Size(ctp_coo.shape)).to(device)  # shape: (num_commodities, num_paths)
        pte_coo = env.paths_to_edges.tocoo()
        self.paths_to_edges = torch.sparse_coo_tensor(
            np.vstack((pte_coo.row, pte_coo.col)),
            torch.DoubleTensor(pte_coo.data),
            torch.Size(pte_coo.shape)).to(device)  # shape: (num_paths, num_edges)
        # self.tm_hist_std = torch.tensor(env.simulator.get_tm_histories_std()).to(device)
        # shape: (num_nodes * (num_nodes - 1),)
        self.edges_capacity = torch.tensor(env.capacity).unsqueeze(1).to(device)  # shape: (num_edges, 1)

    def set_model(self, model):
        self.model = model

    def loss(self, y_pred_batch, y_true_batch):
        """Compute the loss of the model.

        Args:
            y_pred_batch: the split ratios for the candidate paths
            y_true_batch: the true traffic demand and the optimal mlu
        """
        num_nodes = self.env.num_nodes
        losses = []
        loss_vals = []
        batch_size = y_pred_batch.shape[0]
        beta = self.props.beta
        for i in range(batch_size):
            y_pred = y_pred_batch[[i]]
            y_true = y_true_batch[[i]]
            opt = y_true[0][num_nodes * (num_nodes - 1)].item()
            y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))  # shape: (1, num_commodities)

            y_pred = y_pred + 1e-16
            paths_weight = torch.transpose(y_pred, 0, 1)  # shape: (num_paths, 1)
            commodity_total_weight = self.commodities_to_paths.matmul(paths_weight)  # shape: (num_commodities, 1)
            paths_over_total = self.commodities_to_paths.transpose(0, 1).matmul(
                1.0 / commodity_total_weight)  # shape: (num_paths, 1)
            split_ratios = paths_weight.mul(paths_over_total)  # shape: (num_paths, 1)
            tmp_demand_on_paths = self.commodities_to_paths.transpose(0, 1).matmul(
                y_true.transpose(0, 1))  # shape: (num_paths, 1)
            demand_on_paths = tmp_demand_on_paths.mul(split_ratios)  # shape: (num_paths, 1)
            flow_on_edges = self.paths_to_edges.transpose(0, 1).matmul(demand_on_paths)  # shape: (num_edges, 1)
            congestion = flow_on_edges.divide(self.edges_capacity)  # shape: (num_edges, 1)
            mean_cong = torch.mean(congestion.flatten(), dim=0)
            max_cong = torch.max(congestion.flatten(), dim=0).values

            # if self.env.constant_pathlen:
            #     max_sensitivity = torch.max(split_ratios.view(num_nodes * (num_nodes - 1), -1), dim=1).values
            #     # shape: (num_commodities,)
            # else:
            #     split_ratios = torch.split(split_ratios, self.env.commodities_to_path_nums)
            #     max_sensitivity = torch.tensor([torch.max(split_ratio) for split_ratio in split_ratios]).to(self.device)
            #     # shape: (num_commodities,)
            # weight_max_sensitivity = max_sensitivity.mul(self.tm_hist_std) #(num_commodities,)
            # sum_wm_sens = torch.sum(weight_max_sensitivity) / len(weight_max_sensitivity)

            # loss function, the first term is the congestion, the second term is the sensitivity.
            # The operation of dividing by item() is used to balance different objectives,
            # ensuring they are on the same scale. Then, alpha is used to adjust their importance.
            # loss = 1.0 - max_cong if max_cong.item() == 0.0 else \
            #         max_cong / max_cong.item() + self.props.alpha * sum_wm_sens / sum_wm_sens.item()
            # loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong / max_cong.item()
            loss = 1.0 - max_cong if max_cong.item() == 0.0 \
                else (max_cong + mean_cong * beta) / (max_cong + mean_cong * beta).item()
            loss_val = 1.0 if opt == 0.0 else (max_cong + mean_cong * beta).item() / opt
            losses.append(loss)
            loss_vals.append(loss_val)

        ret = sum(losses) / len(losses)
        ret_val = sum(loss_vals) / len(loss_vals)
        return ret, ret_val

    def calculate_split_ratio(self, y_pred):
        y_pred = y_pred + 1e-16
        paths_weight = torch.transpose(y_pred, 0, 1)  # shape: (num_paths, 1)
        commodity_total_weight = self.commodities_to_paths.matmul(paths_weight)  # shape: (num_commodities, 1)
        paths_over_total = self.commodities_to_paths.transpose(0, 1).matmul(
            1.0 / commodity_total_weight)  # shape: (num_paths, 1)
        split_ratios_ori = paths_weight.mul(paths_over_total).to('cpu')  # shape: (num_paths, 1)
        split_ratios = torch.split(split_ratios_ori, self.env.commodities_to_path_nums)
        return split_ratios, split_ratios_ori

    def train(self, train_dl, test_dl, optimizer):
        """Train the model with the given data.

        Args:
            train_dl: the train data loader
            test_dl: the test data loader
            optimizer: the optimizer for the model
        """
        model = self.model
        device = self.device
        model.to(device)
        model_save_name = f"{self.props.topo_name}_{self.props.hist_len}_{self.props.num_layer}" \
                          f"_{self.props.dataset_label}"
        model_save_path = os.path.join(MODEL_DIR, f'{model_save_name}.pt')
        # for epoch in range(self.props.epochs):
        #     with tqdm(train_dl) as tepoch:
        #         for inputs, targets in tepoch:
        #             tepoch.set_description(f"Epoch {epoch + 1}/{self.props.epochs}")
        #             inputs, targets = inputs.to(device), targets.to(device)
        #             preds = model(inputs)
        #             loss, loss_val = self.loss(preds, targets)
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()
        #             tepoch.set_postfix(loss_val=loss_val)
        best_loss = float('inf')
        for epoch in range(self.props.epochs):
            model.train()
            avg_loss = 0
            print("{:-^50}".format(f'Epoch {epoch + 1}/{self.props.epochs}'))
            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = self.model(inputs)
                loss, loss_val = self.loss(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(f'Batch {i + 1}/{len(train_dl)} loss_val: {loss_val}')
                avg_loss += loss_val
            avg_loss /= len(train_dl)
            with open(f'loss_{self.props.dataset_label}_train.log', 'a') as f:
                f.write(f'{avg_loss}\n')

            model.eval()
            avg_loss = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_dl):
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = model(inputs)
                    _, loss_val = self.loss(preds, targets)
                    avg_loss += loss_val
                    # print(f'Batch {i + 1}/{len(test_dl)} loss_val: {loss_val}')
            avg_loss /= len(test_dl)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_save_path)
            with open(f'loss_{self.props.dataset_label}_test.log', 'a') as f:
                f.write(f'{avg_loss}\n')

    def test(self, test_dl):
        """Test the model with the given data.

        Args:
            test_dl: the test data loader
        """
        model = self.model
        device = self.device
        result_save_path = os.path.join(RESULT_DIR, self.props.topo_name, 'Figret', f'result_{self.dataset_label}.txt')
        model.to(device)
        model.eval()
        with torch.no_grad():
            with tqdm(test_dl) as tests:
                test_losses = []
                for inputs, targets in tests:
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = model(inputs)
                    _, loss_val = self.loss(preds, targets)
                    test_losses.append(loss_val)

        print_to_txt(test_losses, result_save_path)

    def inference(self, input_, spine_switch_num):
        """Inference the model with the given data.

        Args:
            input_: single input traffic matrices, shape=(hist_len, num_nodes * num_nodes)
        """
        env = self.env
        device = self.device
        model = self.model
        pod_num = input_.shape[1]
        hist = np.zeros((input_.shape[0], pod_num + spine_switch_num, pod_num + spine_switch_num))
        hist[:, :pod_num, :pod_num] = input_.squeeze()
        hist = hist.reshape(hist.shape[0], hist.shape[1] * hist.shape[2])

        tm_mask = np.ones((env.num_nodes, env.num_nodes), dtype=bool)
        np.fill_diagonal(tm_mask, 0)
        tm_mask = tm_mask.flatten()
        hist = hist[:, tm_mask].flatten()
        hist = normalize_size(hist)
        hist = torch.from_numpy(hist).unsqueeze(0).to(device)
        pred = model(hist)
        split_ratio, split_ratio_ori = self.calculate_split_ratio(pred)
        return split_ratio, split_ratio_ori

    def get_single_loss_and_backword(self, split_ratios, y_true, spine_switch_num):
        """
        Return the loss of the model for a single split_ratio.
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        y_true = y_true.squeeze()
        env = self.env
        device = self.device
        pod_num = y_true.shape[1]
        hist = np.zeros((1, pod_num + spine_switch_num, pod_num + spine_switch_num))
        hist[0, :pod_num, :pod_num] = y_true[0]
        hist = hist.reshape(hist.shape[0], hist.shape[1] * hist.shape[2])
        tm_mask = np.ones((env.num_nodes, env.num_nodes), dtype=bool)
        np.fill_diagonal(tm_mask, 0)
        tm_mask = tm_mask.flatten()
        hist = hist[:, tm_mask].flatten()
        hist = normalize_size(hist)
        hist = torch.from_numpy(hist).unsqueeze(0).to(device)
        y_true = hist

        split_ratios = split_ratios.to(device)
        tmp_demand_on_paths = self.commodities_to_paths.transpose(0, 1).matmul(
            y_true.transpose(0, 1))  # shape: (num_paths, 1)
        demand_on_paths = tmp_demand_on_paths.mul(split_ratios)  # shape: (num_paths, 1)
        flow_on_edges = self.paths_to_edges.transpose(0, 1).matmul(demand_on_paths)  # shape: (num_edges, 1)
        congestion = flow_on_edges.divide(self.edges_capacity)  # shape: (num_edges, 1)
        mean_cong = torch.mean(congestion.flatten(), dim=0)
        max_cong = torch.max(congestion.flatten(), dim=0).values
        beta = self.props.beta
        loss = 1.0 - max_cong if max_cong.item() == 0.0 \
            else (max_cong + mean_cong * beta) / (max_cong + mean_cong * beta).item()
        loss.backward()
        self.optimizer.step()


class FigretDataset(Dataset):
    """Dataset for the FigretNetWork."""

    def __init__(
            self,
            model_name: str,
            hist_len: int,
            pod_num: int,
            spine_num: int,
            hist_names: Union[str, List[str | int]],
            mode: Literal['train', 'test'],
            single_hist_size: int,
    ):
        """
        Initialize the FigretDataset with the history names.
        :param model_name: 模型名称。
        :param hist_len: 训练时设定的hist长度。
        :param pod_num: Pod数量。
        :param spine_num: 单Pod内Spine交换机数量。
        :param hist_names: 如果是字符串，则用逗号分隔多个文件名，如果是列表，则直接使用列表。
                           文件名不包含后缀，现阶段后缀包括.hist, .npy和.npz。
        :param mode: train or test，用于区分训练集和测试集。
        :param single_hist_size: 单个数据文件的长度。
        """
        self.model_name = model_name
        self.hist_len = hist_len
        self.pod_num = pod_num
        self.spine_num = spine_num
        self.hist_names = self.deal_with_hist_names(hist_names)
        self.mode = mode
        self.single_hist_size = single_hist_size

        self.tm_mask = np.ones((pod_num, pod_num), dtype=bool)
        np.fill_diagonal(self.tm_mask, 0)
        self.tm_mask = np.repeat(self.tm_mask[:, :, np.newaxis], spine_num, axis=2)
        self.tm_mask = self.tm_mask.flatten()

        # 用于存放跨hist文件的数据，只有当hist_len大于1时才会用到。
        # 存储的方式是：key为hist_idx，value为hist_idx对应的头尾数据列表。
        # 如果hist_len为2，那么value长度为4，其中分别是当前hist的前2个数据和后2个数据。
        self.head_tail_data = defaultdict(list)

        dataset_len = 0
        # 获取所有hist文件的数据长度.
        for hist in self.hist_names:
            try:
                data = np.load(f'{DATA_DIR}/{model_name}/{mode}/{hist}.npz')
                shape = data['shape']
                dataset_len += shape[0]
            except FileNotFoundError:
                data = self._cached_data(hist)
                dataset_len += data.shape[0] + self.hist_len - 1
        self.dataset_len = dataset_len - self.hist_len

        self.opts = []
        self.load_opt()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        hist_idx = idx // self.single_hist_size
        offset = idx % self.single_hist_size
        if self.single_hist_size - offset > self.hist_len - 1:
            X_data = self._cached_data(hist_idx)[offset]
        else:
            # 确保需要的首尾数据已经缓存。
            if hist_idx not in self.head_tail_data:
                self._cached_data(hist_idx)
            if hist_idx + 1 not in self.head_tail_data:
                self._cached_data(hist_idx + 1)
            # 计算当前是在下一个hist中的第几个位置。如果是1，说明需要下一个hist的第一个数据，
            # 和之前hist的hist_len - 1个数据。
            next_hist_offset = offset - self.single_hist_size + self.hist_len
            X_data = []
            for i in range(self.hist_len - next_hist_offset):
                X_data.append(self.head_tail_data[hist_idx][i - self.hist_len + 1])
            for i in range(next_hist_offset):
                X_data.append(self.head_tail_data[hist_idx + 1][i])
            X_data = np.array(X_data).flatten()
        tm_len = self.pod_num * (self.pod_num - 1) * self.spine_num
        offset += 1
        if self.single_hist_size - offset > self.hist_len - 1:
            next_tm = self._cached_data(hist_idx)[offset][-tm_len:]
        else:
            if hist_idx not in self.head_tail_data:
                self._cached_data(hist_idx)
            if hist_idx + 1 not in self.head_tail_data:
                self._cached_data(hist_idx + 1)
            next_tm = self.head_tail_data[hist_idx + 1][offset - self.single_hist_size + self.hist_len - 1]
        Y_data = np.append(next_tm, self.opts[idx + self.hist_len]).reshape(1, -1).squeeze()
        return X_data, Y_data

    def load_opt(self):
        """Load the optimal MLU from the file."""
        for hist in self.hist_names:
            with open(f'{DATA_DIR}/{self.model_name}/{self.mode}/{hist}.opt', 'r') as f:
                for line in f:
                    self.opts.append(float(line.strip()))

    @lru_cache(maxsize=6)
    def _cached_data(self, hist_idx):
        hist_name = os.path.join(DATA_DIR, self.model_name, 'train', str(self.hist_names[hist_idx]))
        if os.path.exists(f'{hist_name}.npz'):
            print(f'Loading {hist_name}.npz ...')
            hist = np.load(f'{hist_name}.npz')
            hist = hist['arr_0'].squeeze()
        elif os.path.exists(f'{hist_name}.npy'):
            print(f'Loading {hist_name}.npy ...')
            hist = np.load(f'{hist_name}.npy').squeeze()
            shape = np.array(hist.shape)
            np.savez_compressed(f'{hist_name}.npz', arr_0=hist, shape=shape)
        else:
            hist = []
            print(f'Loading {hist_name}.hist ...')
            with open(f'{hist_name}.hist', 'r') as f:
                for line in f:
                    hist.append(np.array(line.strip().split(' ')).astype(np.float64))
            hist = np.array(hist)
            shape = np.array(hist.shape)
            np.savez_compressed(f'{hist_name}.npz', arr_0=hist, shape=shape)
        hist = normalize_size(hist)
        # 读入的hist shape是(n, pod_num * pod_num * spine_num)
        # 需要处理成n, pod_num * spine_num * (pod_num - 1)形式
        hist = hist.reshape(hist.shape[0], self.pod_num, self.pod_num, self.spine_num)
        hist = hist.transpose(0, 1, 3, 2).reshape(hist.shape[0], self.pod_num * self.pod_num * self.spine_num)
        hist = hist[:, self.tm_mask]

        # 根据hist_len叠加数据。
        n, m = hist.shape
        X = np.zeros((n - self.hist_len + 1, self.hist_len * m))
        for i in range(self.hist_len):
            X[:, i * m:(i + 1) * m] = hist[i:n - self.hist_len + i + 1, :]

        # 存储头尾数据，头尾部分的叠加的数据只能通过当前hist和其他hist共同叠加得到。
        for i in range(self.hist_len - 1):
            self.head_tail_data[hist_idx].append(np.copy(hist[i]))
        for i in range(self.hist_len - 1):
            self.head_tail_data[hist_idx].append(np.copy(hist[i + 1 - self.hist_len]))
        return X

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
