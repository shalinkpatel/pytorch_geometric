from typing import Optional

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

class PGExplainer(torch.nn.Module):
    pass