import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class ProblemPool(Dataset):
    """
    一个problem由两部分组成，original_state和newly_added。

    - original_state是一个01二维数组，表示一个pod内的原始流量需求。每一列代表一个spine，每一行代表一个leaf。
      original_state[i][j] = 1 表示第i个leaf需要连接到第j个spine。

    - newly_added是一个一维数组，长度等于spine的数量，表示新添加的spine的流量需求。
      newly_added[i] = n 表示第i个spine需要添加n个到其他leaf的流量。

    - 合法性保证：对于每个spine，剩余空闲的到leaf的流量需求大于等于newly_added[i]。

    - answer是一个01二维数组，表示添加过后的pod内流量需求。解决这个问题的目标是使得添加的流量需求数量与newly_added一致。
    """

    def __init__(self, leaf_num, spine_num, data_path, dataset_name, with_link_weight):
        # 先假定leaf_num和spine_num相等
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.with_link_weight = with_link_weight
        if dataset_name is None:
            dataset_name = f'l{leaf_num}-s{spine_num}.csv'
        if with_link_weight:
            self.file = f"{data_path}/server/with-link-weight/{dataset_name}"
        else:
            self.file = f"{data_path}/server/no-link-weight/{dataset_name}"
        if not os.path.exists(self.file):
            with open(self.file, 'w') as f:
                f.write("")
            self.df = pd.DataFrame()
        else:
            self.df = pd.read_csv(self.file, header=None)
        self.problem_count = len(self.df)

    def __len__(self):
        return self.problem_count

    def __getitem__(self, idx):
        if idx >= self.problem_count or idx < 0:
            raise IndexError("Index out of range")
        return torch.tensor(self.df.iloc[idx:idx + 1, :].values, dtype=torch.float32)

    def random_select(self):
        if self.problem_count == 0:
            return None
        idx = np.random.randint(0, self.problem_count - 1)
        return self[idx]

    @property
    def original_state(self):
        original_state = self.df.iloc[:, 2 + self.leaf_num: 2 + self.leaf_num + self.leaf_num * self.spine_num]
        original_state = original_state.values.reshape(original_state.shape[0], self.leaf_num, self.spine_num)
        return original_state

    @property
    def link_weight(self):
        link_weight = self.df.iloc[:, 2 + self.leaf_num + self.leaf_num * self.spine_num:]
        link_weight = link_weight.values.reshape(link_weight.shape[0], self.leaf_num, self.spine_num)
        return link_weight

    def _generate_problem(self):
        """
        随机生成一个Problem
        """
        # 生成original_state
        ratio_0_1 = np.random.uniform(0.2, 0.8)
        random_variable = {0: 0.99 * ratio_0_1, 1: 0.99 * (1 - ratio_0_1), 2: 0.008, 3: 0.0015, 4: 0.0005}
        original_state = np.random.choice(list(random_variable.keys()), size=(self.leaf_num, self.spine_num),
                                          p=list(random_variable.values()))

        # 生成link_weight
        link_weight = np.random.randint(0, 256, (self.leaf_num, self.spine_num))
        link_weight = link_weight / 255

        # 可能合法的leaf
        use_or_not = (original_state >= 1).astype(int)
        used_leaf_links = np.sum(use_or_not, axis=1)
        legal_leaf_flag = (used_leaf_links < self.spine_num).astype(int)
        choices = np.where(legal_leaf_flag == 1)[0]

        # 生成M和N
        leaf_spine_requirement = [0] * 2
        leaf_spine_requirement[0] = np.random.randint(1, len(choices) + 1)
        leaf_spine_requirement[1] = np.random.randint(1, self.spine_num)

        # 选出非法leaf
        # probs = (1 - used_leaf_links[choices] / self.spine_num) * 0.8
        # probs = probs / np.sum(probs)
        if len(choices) - leaf_spine_requirement[0] > 0:
            illegal_num = np.random.randint(0, len(choices) - leaf_spine_requirement[0])
            illegal_choices = np.random.choice(choices, illegal_num, replace=False)
            legal_leaf_flag[illegal_choices] = 0

        leaf_spine_requirement = np.array(leaf_spine_requirement, dtype=np.float32)
        original_state = original_state.flatten()
        link_weight = link_weight.flatten()
        if self.with_link_weight:
            problem = np.concatenate(
                (leaf_spine_requirement, legal_leaf_flag.astype(np.float32), original_state, link_weight.astype(np.float32))
            )
        else:
            problem = np.concatenate(
                (leaf_spine_requirement, legal_leaf_flag.astype(np.float32), original_state)
            )
        self.problem_count += 1
        return problem

    def generate_problems(self, num):
        chunk_size = 50
        for _ in range(num // chunk_size):
            problems = np.array([self._generate_problem() for _ in range(chunk_size)])
            problems_df = pd.DataFrame(problems, dtype=np.float32)
            problems_df.to_csv(self.file, mode='a', header=False, index=False)
            self.df = pd.concat([self.df, problems_df], ignore_index=True)
        if num % chunk_size == 0:
            return
        problems = np.array([self._generate_problem() for _ in range(num % chunk_size)])
        problems_df = pd.DataFrame(problems, dtype=np.float32)
        problems_df.to_csv(self.file, mode='a', header=False, index=False)
        self.df = pd.concat([self.df, problems_df], ignore_index=True)


class LSTMProblemPool(ProblemPool):
    def __init__(self, leaf_num, spine_num, data_path, dataset_name, with_link_weight, sequence_length):
        # 先假定leaf_num和spine_num相等

        super().__init__(leaf_num, spine_num, data_path, dataset_name, with_link_weight)
        self.sequence_length = sequence_length
        if len(self.df) % sequence_length != 0:
            self.df = self.df.iloc[:-(len(self.df) % sequence_length), :]
        self.problem_count = len(self.df) // sequence_length

    def __getitem__(self, idx):
        if idx >= self.problem_count or idx < 0:
            raise IndexError("Index out of range")
        return torch.tensor(self.df.iloc[idx * self.sequence_length:(idx + 1) * self.sequence_length, :].values, dtype=torch.float32)

    def random_select(self):
        if self.problem_count == 0:
            return None
        idx = np.random.randint(0, self.problem_count - 1)
        curr_seq = self[idx]
        idx = np.random.randint(0, curr_seq.shape[0] - 1)
        curr_problem = curr_seq[idx]
        leaf_spine_requirement = curr_problem[:2]
        legal_leaf_flag = curr_problem[2:2 + self.leaf_num]
        original_state = curr_problem[2 + self.leaf_num: 2 + self.leaf_num + self.leaf_num * self.spine_num]
        if self.with_link_weight:
            link_weight = curr_problem[2 + self.leaf_num + self.leaf_num * self.spine_num:]
            return leaf_spine_requirement, legal_leaf_flag, original_state, link_weight
        return leaf_spine_requirement, legal_leaf_flag, original_state


if __name__ == '__main__':
    problem_pool = ProblemPool(32, 16, './problems', True)
    problem_pool.generate_problems(10000)
    print(len(problem_pool))
