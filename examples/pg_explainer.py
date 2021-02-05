import os.path as osp

from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, PGExplainer

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.embed(x, edge_index)
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def embed(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x, edge_index = data.x, data.edge_index

pbar = tqdm(range(200))
for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    log_logits = torch.log(model(x, edge_index))
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    pbar.set_description("Loss -> %.4f" % loss.item())

node_idx = 10
explainer = PGExplainer(model)
explainer.train_explainer(x, edge_index)
edge_mask = explainer.explain(node_idx)
ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, data.y)
plt.show()