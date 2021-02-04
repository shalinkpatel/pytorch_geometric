from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

class PGExplainer(torch.nn.Module):
    def __init__(self, model, epcohs: int = 100, lr: float = 0.01, 
                num_hops: Optional[int] = None, temp: float = 1.0,
                K: int = 5, log: bool = True):
        super(PGExplainer, self).__init__()
        self.model = model
        self.epochs = epcohs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.temp = temp
        self.K = K
        self.log = log

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def __build_explainer__(self, x, edge_index):
        self.Z = self.model.embed(x, edge_index)
        self.Y = self.model.call(x, edge_index)

        self.explainer = nn.Sequential(nn.Linear(3 * self.Z.size(1), 64,
                                    nn.ReLU(),
                                    nn.Linear(64, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 1))).to(x.device)

    def __get_params__(self, edge_index, node_idx):
        z_i = self.Z[edge_index[0, :], :]
        z_j = self.Z[edge_index[1, :], :]
        z_v = self.Z[node_idx, :].tile(1, edge_index.size(1))

        return self.explainer(torch.cat([z_i, z_j, z_v])).reshape(-1)

    def __set_masks__(self, omega):
        eps = nn.Parameter(torch.rand(omega.size(), device=omega.device))
        self.edge_mask = F.sigmoid((torch.log(eps) - torch.log(1 - eps) 
            + omega) / self.temp)
        
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
    
    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __loss__(self, node_idx, edge_index, y_pred_sub):
        loss = -(self.Y[node_idx, :] * torch.log(y_pred_sub)).sum()

        # Graph sparsity constraint

        # Connectivity constraint


    def train_explainer(self, x, edge_index, **kwargs):
        self.model.eval()
        self.to(x.device)

        # Initial prediction and we create the explainer network
        with torch.no_grad():
            self.__build_explainer__(x, edge_index)

        optimizer = torch.optim.Adam(self.explainer.parameters(),
                                     lr=self.lr)

        self.comp_graphs = []
        self.x_neigh = []
        self.node_mapping = []
        for node in range(x.size(0)):
            # Each node operates on a k-hop subgraph
            x, edge_index_sub, mapping, _, _ = self.__subgraph__(
                node, x, edge_index, **kwargs)
            self.comp_graphs.append(edge_index_sub)
            self.x_neigh.append(x)
            self.node_mapping.append(mapping)

        if self.log:
            pbar = tqdm(range(self.epochs))
        else:
            pbar = range(self.epochs)

        for _ in pbar:
            optimizer.zero_grad()
            loss = 0
            for node in range(x.size(0)):
                omega = self.__get_params__(self.comp_graphs[node], node)
                for _ in range(self.K):
                    # Draw the edge mask and compute the MI loss
                    self.__set_masks__(omega)
                    y_pred_sub = self.model.call(self.x_neigh[node],
                        self.comp_graphs[node])[self.node_mapping[node]]
                    loss += self.__loss__(node, self.comp_graphs[node], 
                        y_pred_sub)

            loss /= self.K * x.size(0)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.set_description("Loss: %.4f" % loss.item())
        
        self.__clear_masks__()

    def explain(self, node_idx):
        edge_index = self.comp_graphs[node_idx]
        edge_mask = self.__get_params__(edge_index, node_idx)
        return edge_mask, edge_index