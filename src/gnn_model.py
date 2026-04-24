import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FraudGAT(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_heads=2):
        super(FraudGAT, self).__init__()
        
        # First Graph Attention Layer
        # We use edge attributes (transaction features) to help calculate attention
        self.conv1 = GATConv(
            in_channels=node_in_dim, 
            out_channels=hidden_dim, 
            heads=num_heads, 
            edge_dim=edge_in_dim,
            add_self_loops=False
        )
        
        # Second Graph Attention Layer compressing down to a 1D output for classification
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim, 
            heads=1, 
            edge_dim=edge_in_dim,
            add_self_loops=False
        )
        
        # Edge Classification Layer (Source Node + Dest Node + Edge Features -> 1 Logit)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, extract_features=False):
        # 1. Update Node Embeddings based on their neighbors
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # 2. Extract edge source and destination nodes
        src, dst = edge_index
        
        # 3. Concatenate (Source Node Emb + Dest Node Emb + Transaction Features)
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        
        if extract_features:
            # Return the 16-dimensional hidden network representation
            hidden = self.edge_classifier[0](edge_repr)
            hidden = self.edge_classifier[1](hidden)
            return hidden
            
        # 4. Predict if the edge (transaction) is fraud
        return self.edge_classifier(edge_repr).squeeze()
