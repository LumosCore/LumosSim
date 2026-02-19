from torch import nn


class FigretNetWork(nn.Module):
    def __init__(self, hist_len, pod_num, spine_num, layer_num):
        """Initialize the FigretNetWork with the network structure.

        Args:
            hist_len: length of the history traffic matrix
            pod_num: number of pods
            spine_num: number of spines per pod
            layer_num: number of hidden layers
        """
        super(FigretNetWork, self).__init__()
        self.input_dim = hist_len * pod_num * spine_num * (pod_num - 1)
        self.output_dim = pod_num * spine_num * (pod_num - 1)
        self.pod_num = pod_num
        self.spine_num = spine_num
        self.flatten = nn.Flatten()
        self.layers = []
        self.layers.append(nn.Linear(self.input_dim, 256))
        self.layers.append(nn.ReLU())
        for _ in range(layer_num):
            self.layers.append(nn.Linear(256, 256))
            self.layers.append(nn.ReLU())
            self.dropout = nn.Dropout(p=0.4)
        self.layers.append(nn.Linear(256, self.output_dim))
        self.net = nn.Sequential(*self.layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Forward the input data through the network.

        Args:
            x: input data, history len * flattened traffic matrix
        """
        x = self.flatten(x)
        x = self.net(x)
        x = x.view(x.size(0), self.pod_num, self.spine_num, self.pod_num - 1)
        x = self.softmax(x)
        x = x.view(x.size(0), -1)
        return x
