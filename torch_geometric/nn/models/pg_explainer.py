from typing import Optional

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

class PGExplainer(torch.nn.Module):
    def __init__(self, embed_model, output_model, epcohs: int = 100, 
                lr: float = 0.01, num_hops: Optional[int] = None, 
                log: bool = True):
        super(PGExplainer, self).__init__()
        self.embed_model = embed_model
        self.output_model = output_model
        self.epochs = epcohs
        self.lr = lr
        self.__num_hops__ = num_hops
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

    
