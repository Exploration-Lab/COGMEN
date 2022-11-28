import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GNN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=args.gnn_nheads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * args.gnn_nheads)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index)))

        return x
