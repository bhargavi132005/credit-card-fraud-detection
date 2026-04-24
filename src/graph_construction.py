import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import numpy as np


def create_graph_data(df: pd.DataFrame):
    """
    Converts a transaction DataFrame into a PyTorch Geometric Data object.

    Nodes: Unique accounts ('nameOrig', 'nameDest').
    Edges: Transactions.
    Edge Features: Scaled transaction features.
    Edge Labels: 'isFraud' flag.
    """
    print("\nConstructing graph...")

    # Create a unified mapping for all unique accounts
    all_accounts = pd.concat([df['nameOrig'], df['nameDest']]).unique()
    account_map = {name: i for i, name in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    # Map source and destination accounts to their integer indices
    src = df['nameOrig'].map(account_map).values
    dst = df['nameDest'].map(account_map).values

    # Create edge_index tensor
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # Define edge features and labels
    edge_label = torch.tensor(df['isFraud'].values, dtype=torch.long)

    # Select and scale edge features
    edge_feature_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    # Ensure all columns exist, fill missing with 0 (for types not in sample)
    for col in edge_feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    scaler = StandardScaler()
    edge_features = scaler.fit_transform(df[edge_feature_cols].values)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Create initial node features. Instead of a memory-intensive identity matrix
    # (which would be num_nodes x num_nodes), we create a smaller, learnable
    # feature vector for each node. The GNN will update these during training.
    node_feature_dim = 16
    node_features = torch.randn(num_nodes, node_feature_dim, dtype=torch.float)

    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_label)
    print("Graph construction complete.")
    print(graph_data)

    return graph_data, account_map