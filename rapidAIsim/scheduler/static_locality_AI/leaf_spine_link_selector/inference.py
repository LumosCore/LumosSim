import numpy as np
from .model import Model, LSTMModel
import torch
from .train import Trainer
import os


class Inference:
    def __init__(self, leaf_num, spine_num, root_path, with_link_weight, model_path=None):
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_path = root_path
        self.with_link_weight = with_link_weight
        self.model = self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path):
        leaf_num, spine_num = self.leaf_num, self.spine_num
        if self.with_link_weight:
            model = Model(leaf_num * spine_num * 2 + 2 + leaf_num, leaf_num + spine_num)
        else:
            model = Model(leaf_num * spine_num + 2 + leaf_num, leaf_num + spine_num)
        if model_path is None:
            if self.with_link_weight:
                model_path = os.path.join(
                    self.root_path, "saved_models/with-link-weight/model-l{}-s{}.pth".format(leaf_num, spine_num))
            else:
                model_path = os.path.join(
                    self.root_path, "saved_models/no-link-weight/model-l{}-s{}.pth".format(leaf_num, spine_num))
        model.load_state_dict(torch.load(model_path))
        return model

    def inference(self, leaf_spine_requirement, original_state, valid_leaves, link_weight):
        if self.with_link_weight:
            if np.max(np.max(link_weight)) > 0:
                link_weight = link_weight / np.max(np.max(link_weight))
            input_ = torch.tensor(
                [*leaf_spine_requirement, *valid_leaves, *original_state.flatten(), *link_weight.flatten()],
                dtype=torch.float32)
        else:
            input_ = torch.tensor(
                [*leaf_spine_requirement, *valid_leaves, *original_state.flatten()], dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_)

        return self.get_output(input_, output)

    def get_output(self, input_, output):
        loss, indices_leaf, indices_spine = Trainer.get_single_loss_and_output(
            input_, output, self.leaf_num, self.spine_num,
            guarantee_legal=True, with_link_weight=self.with_link_weight)

        indices_leaf = indices_leaf.cpu().numpy()
        indices_spine = indices_spine.cpu().numpy()
        x_i = {f'{i}': 0 for i in range(self.leaf_num)}
        for i in indices_leaf:
            x_i[f'{i}'] = 1
        c_i_j = {f'{i}_{j}': 0 for i in range(self.leaf_num) for j in range(self.spine_num)}
        for i in indices_leaf:
            for j in indices_spine:
                c_i_j[f'{i}_{j}'] = 1
        return c_i_j, x_i, loss.numpy()


class LSTMInference(Inference):
    def __init__(self, leaf_num, spine_num, root_path, with_link_weight, lstm_hidden_dim, lstm_layers, model_path=None):
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        super().__init__(leaf_num, spine_num, root_path, with_link_weight, model_path)
        self.model.eval()
        self.hidden = torch.zeros(lstm_layers, 1, lstm_hidden_dim).to(self.device)
        self.cell_state = torch.zeros(lstm_layers, 1, lstm_hidden_dim).to(self.device)

    def load_model(self, model_path):
        leaf_num, spine_num = self.leaf_num, self.spine_num
        if self.with_link_weight:
            model = LSTMModel(
                leaf_num * spine_num * 2 + 2 + leaf_num, leaf_num + spine_num, self.lstm_hidden_dim, self.lstm_layers)
        else:
            model = LSTMModel(
                leaf_num * spine_num + 2 + leaf_num, leaf_num + spine_num, self.lstm_hidden_dim, self.lstm_layers)
        if model_path is None:
            if self.with_link_weight:
                model_path = os.path.join(
                    self.root_path, "saved_models/with-link-weight/model-l{}-s{}-lstm.pth".format(leaf_num, spine_num))
            else:
                model_path = os.path.join(
                    self.root_path, "saved_models/no-link-weight/model-l{}-s{}-lstm.pth".format(leaf_num, spine_num))
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        return model

    def inference(self, leaf_spine_requirement, original_state, valid_leaves, link_weight):
        if self.with_link_weight:
            link_weight /= 256
            input_ = torch.tensor(
                [*leaf_spine_requirement, *valid_leaves, *original_state.flatten(), *link_weight.flatten()],
                dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            input_ = torch.tensor(
                [*leaf_spine_requirement, *valid_leaves, *original_state.flatten()],
                dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output, (self.hidden, self.cell_state) = self.model(input_.to(self.device), (self.hidden, self.cell_state))
        self.hidden.detach()
        self.cell_state.detach()

        return self.get_output(input_[0][0], output[0][0])
