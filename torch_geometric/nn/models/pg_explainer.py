from copy import copy
from math import sqrt
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

EPS = 1e-15


class PGExplainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.05,
    }

    def __init__(self, model, epcohs: int = 30, lr: float = 0.01, 
                num_hops: Optional[int] = None, temp: float = 3.0,
                K: int = 5, budget: float = -1.0, log: bool = True):
        super(PGExplainer, self).__init__()
        self.model = model
        self.epochs = epcohs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.temp = temp
        self.K = K
        self.budget = budget
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
        self.Y = self.model(x, edge_index)

        self.explainer = nn.Sequential(nn.Linear(3 * self.Z.size(1), 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 1)).to(x.device)

    def __get_params__(self, edge_index, node_idx):
        z_i = self.Z[edge_index[0, :], :]
        z_j = self.Z[edge_index[1, :], :]
        z_v = self.Z[node_idx, :].repeat(edge_index.size(1), 1)

        return self.explainer(torch.cat([z_i, z_j, z_v], dim=1)).reshape(-1)

    def __set_masks__(self, omega):
        eps = torch.rand(omega.size(), device=omega.device)
        self.edge_mask = (torch.log(eps) - torch.log(1 - eps) 
            + omega) / self.temp
        
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
        loss = -torch.log(y_pred_sub[torch.argmax(self.Y[node_idx, :])]  + EPS)

        # Graph sparsity constraint
        if self.budget > 0:
            size_loss = F.relu((self.edge_mask - torch.tile(self.budget, 
                (self.edge_mask.size(0),))).sum())
        else:
            size_loss = self.edge_mask.sum()

        # Connectivity constraint
        # Figure out later

        return loss + self.coeffs['edge_size'] * size_loss


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
            x_sub, edge_index_sub, mapping, _, _ = self.__subgraph__(
                node, x, edge_index, **kwargs)
            self.comp_graphs.append(edge_index_sub)
            self.x_neigh.append(x_sub)
            self.node_mapping.append(mapping.item())

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
                    y_pred_sub = self.model(self.x_neigh[node],
                        self.comp_graphs[node])[self.node_mapping[node], :]
                    loss += self.__loss__(node, self.comp_graphs[node], 
                        y_pred_sub)

            loss /= self.K * x.size(0) ** 2
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.set_description("Loss -> %.4f" % loss.item())
        
        self.__clear_masks__()

    def explain(self, node_idx):
        edge_index = self.comp_graphs[node_idx]
        edge_mask = self.__get_params__(edge_index, node_idx)
        return torch.sigmoid(edge_mask)

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, _ = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def __repr__(self):
        return f'{self.__class__.__name__}()'