import os.path

from model import Model, LSTMModel
from problem_pool import ProblemPool, LSTMProblemPool
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


class Trainer:
    def __init__(self, leaf_num, spine_num, epoch_num, lr, batch_size, train_ratio, root_path, dataset_name=None,
                 with_link_weight=False):
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.epoch_num = epoch_num
        self.lr = lr
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.root_path = root_path
        self.with_link_weight = with_link_weight

        self.kf = KFold(n_splits=5, shuffle=True, random_state=24)

        self.problem_pool = ProblemPool(leaf_num, spine_num, os.path.join(root_path, 'problems'), dataset_name,
                                        with_link_weight)
        train_size = int(len(self.problem_pool) * train_ratio)
        test_size = len(self.problem_pool) - train_size
        train_set, test_set = torch.utils.data.random_split(self.problem_pool, [train_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if with_link_weight:
            self.model = Model(leaf_num * spine_num * 2 + 2 + leaf_num, leaf_num + spine_num)
        else:
            self.model = Model(leaf_num * spine_num + 2 + leaf_num, leaf_num + spine_num)

        # self.model = Model(leaf_num, leaf_num)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, verbose=True):
        for epoch in range(self.epoch_num):
            self.model.train()
            for i, inputs in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                # inputs = inputs[:, :, 2:2 + self.leaf_num]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                losses = self.get_loss(inputs, outputs)
                losses.backward()
                self.optimizer.step()
                if verbose:
                    print(f"Epoch {epoch}, batch {i}, loss: {losses.item()}")

            self.model.eval()
            with torch.no_grad():
                for i, inputs in enumerate(self.test_loader):
                    # inputs = inputs[:, :, 2:2 + self.leaf_num]
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    losses = self.get_loss(inputs, outputs)
                    if verbose:
                        print(f"Epoch {epoch}, test batch {i}, loss: {losses.item()}")

        # Save model
        if self.with_link_weight:
            save_path = os.path.join(self.root_path, 'saved_models', 'with-link-weight')
        else:
            save_path = os.path.join(self.root_path, 'saved_models', 'no-link-weight')
        torch.save(self.model.state_dict(), f"{save_path}/model-l{self.leaf_num}-s{self.spine_num}.pth")
        return losses.item()

    def get_loss(self, inputs, outputs):
        losses = []
        for i in range(outputs.shape[0]):
            loss, _, _ = self.get_single_loss_and_output(inputs[i][0], outputs[i][0], self.leaf_num, self.spine_num,
                                                         with_link_weight=self.with_link_weight)
            losses.append(loss.to(self.device))
        losses = torch.stack(losses)
        losses = torch.mean(losses)
        # losses.requires_grad = True
        return losses

    @staticmethod
    def get_single_loss_and_output(input_, output, leaf_num, spine_num, guarantee_legal=False, with_link_weight=False):
        input_ = input_.cpu()
        output = output.cpu()
        leaf_output = output[:leaf_num]
        spine_output = output[leaf_num:]
        leaf_flag = input_[2:2 + leaf_num]
        leaf_flag = torch.diag(leaf_flag)
        leaf_output = torch.matmul(leaf_output.reshape(1, leaf_num), leaf_flag)
        original_state = input_[2 + leaf_num:2 + leaf_num + leaf_num * spine_num].reshape(leaf_num, spine_num)
        if with_link_weight:
            link_weight = input_[2 + leaf_num + leaf_num * spine_num:].reshape(leaf_num, spine_num)
            original_state = torch.multiply(original_state, link_weight)
        if guarantee_legal:
            leaf_output[0] = leaf_output[0] + input_[2:2 + leaf_num]

        probs_leaf, indices_leaf = torch.topk(leaf_output[0], input_[0].int().item())
        probs_spine, indices_spine = torch.topk(spine_output, input_[1].int().item())

        assert all(input_[2:2 + leaf_num][indices_leaf])

        if guarantee_legal:
            probs_leaf -= 1

        selected_leaves = torch.zeros((1, leaf_num))
        for i in range(len(indices_leaf)):
            selected_leaves[0, indices_leaf[i]] = probs_leaf[i]
        selected_spines = torch.zeros((spine_num, 1))
        for i in range(len(indices_spine)):
            selected_spines[indices_spine[i], 0] = probs_spine[i]
        loss1 = torch.sum(torch.matmul(selected_leaves, torch.matmul(original_state, selected_spines)))
        # # 使用交叉熵损失函数将loss1与input_[0][2:2 + self.leaf_num]进行比较
        # leaf_flag = torch.zeros(self.leaf_num, dtype=torch.float32)
        # for i in indices_leaf:
        #     leaf_flag[i] = input_[0][2 + i]
        # loss2 = nn.functional.binary_cross_entropy(selected_leaves[0], leaf_flag)
        # loss2 = nn.functional.binary_cross_entropy(leaf_output, input_[0][2:2 + self.leaf_num])
        # loss2 = nn.functional.binary_cross_entropy(leaf_output, input_[0])
        # loss = loss1 + loss2 * 10
        # print(selected_leaves[0], "\n", leaf_flag)
        return loss1, indices_leaf, indices_spine


class LSTMTrainer:
    def __init__(self, leaf_num, spine_num, epoch_num, lr, batch_size, train_ratio, root_path, dataset_name=None,
                 with_link_weight=False, sequence_length=50, lstm_hidden_dim=128, lstm_layers=2):
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.epoch_num = epoch_num
        self.lr = lr
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.root_path = root_path
        self.with_link_weight = with_link_weight
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        self.problem_pool = LSTMProblemPool(leaf_num, spine_num, os.path.join(root_path, 'problems'), dataset_name,
                                            with_link_weight, sequence_length=sequence_length)
        train_size = int(len(self.problem_pool) * train_ratio)
        test_size = len(self.problem_pool) - train_size
        train_set, test_set = torch.utils.data.random_split(self.problem_pool, [train_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if with_link_weight:
            feature_dim = leaf_num * spine_num * 2 + 2 + leaf_num
            self.model = LSTMModel(feature_dim, leaf_num + spine_num, lstm_hidden_dim, lstm_layers)
        else:
            feature_dim = leaf_num * spine_num + 2 + leaf_num
            self.model = LSTMModel(feature_dim, leaf_num + spine_num, lstm_hidden_dim, lstm_layers)

        # self.model = Model(leaf_num, leaf_num)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, verbose=True):
        for epoch in range(self.epoch_num):
            self.model.train()
            for i, inputs in enumerate(self.train_loader):
                hidden = self.init_hidden(self.lstm_layers, inputs.shape[0], self.lstm_hidden_dim)
                self.optimizer.zero_grad()
                # inputs = inputs[:, :, 2:2 + self.leaf_num]
                if self.with_link_weight:
                    inputs[:, :, 2 + self.leaf_num + self.leaf_num * self.spine_num:] /= 256
                    inputs[:, :, 2 + self.leaf_num + self.leaf_num * self.spine_num:] += 1e-5
                else:
                    inputs = inputs[:, :, :2 + self.leaf_num + self.leaf_num * self.spine_num]
                inputs = inputs.to(self.device)
                outputs, hidden = self.model(inputs, hidden)
                losses = self.get_loss(inputs, outputs)
                losses.backward()
                self.optimizer.step()
                if verbose:
                    print(f"Epoch {epoch}, batch {i}, loss: {losses.item()}")

            self.model.eval()
            with torch.no_grad():
                for i, inputs in enumerate(self.test_loader):
                    # inputs = inputs[:, :, 2:2 + self.leaf_num]
                    hidden = self.init_hidden(self.lstm_layers, inputs.shape[0], self.lstm_hidden_dim)
                    if self.with_link_weight:
                        inputs[:, :, 2 + self.leaf_num + self.leaf_num * self.spine_num:] /= 256
                    else:
                        inputs = inputs[:, :, :2 + self.leaf_num + self.leaf_num * self.spine_num]
                    inputs = inputs.to(self.device)
                    outputs, hidden = self.model(inputs, hidden)
                    losses = self.get_loss(inputs, outputs)
                    if verbose:
                        print(f"Epoch {epoch}, test batch {i}, loss: {losses.item()}")

        # Save model
        if self.with_link_weight:
            save_path = os.path.join(self.root_path, 'saved_models', 'with-link-weight')
        else:
            save_path = os.path.join(self.root_path, 'saved_models', 'no-link-weight')
        torch.save(self.model.state_dict(), f"{save_path}/model-l{self.leaf_num}-s{self.spine_num}-lstm.pth")
        return losses.item()

    def get_loss(self, inputs, outputs):
        losses = []
        for i in range(outputs.shape[0]):
            for j in range(self.sequence_length):
                loss, _, _ = Trainer.get_single_loss_and_output(inputs[i][j], outputs[i][j], self.leaf_num,
                                                                self.spine_num, with_link_weight=self.with_link_weight)
                losses.append(loss.to(self.device))
        losses = torch.stack(losses)
        losses = torch.mean(losses)
        # losses.requires_grad = True
        return losses

    def init_hidden(self, num_layers, batch_size, hidden_dim):
        return (torch.zeros(num_layers, batch_size, hidden_dim).to(self.device),
                torch.zeros(num_layers, batch_size, hidden_dim).to(self.device))


if __name__ == '__main__':

    trainer = LSTMTrainer(32, 16, 150, 0.001, 16, 0.8,
                          '.', 'l32-s16-real.csv', False, 20, 128, 2)
    trainer.train()
