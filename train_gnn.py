import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import os

from src.gnn_model import FraudGAT


def main():
    # 1. Load the saved graph data
    graph_path = "results/graph_data.pt"
    if not os.path.exists(graph_path):
        print("Graph data not found! Please run main.py first.")
        return

    print("Loading graph data...")
    data = torch.load(graph_path, weights_only=False)
    
    # 2. Train/Test Split on Edges (Transactions)
    # We will use 80% of transactions to train, and 20% to test
    num_edges = data.edge_index.size(1)
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 3. Handle Imbalance (Crucial for GNNs)
    # Calculate how many negative vs positive edges we have in the training set
    num_neg = (data.y[train_indices] == 0).sum().item()
    num_pos = (data.y[train_indices] == 1).sum().item()
    
    # pos_weight forces the loss function to care heavily about the rare fraud cases
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float)
    print(f"\nCalculated Imbalance Weight (pos_weight): {pos_weight.item():.2f}")

    # 4. Initialize the Model, Optimizer, and Loss
    node_in_dim = data.x.size(1)
    edge_in_dim = data.edge_attr.size(1)
    hidden_dim = 16
    
    model = FraudGAT(node_in_dim, edge_in_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 5. Training Loop
    epochs = 100
    print("\nStarting GNN Training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (computes predictions for ALL edges)
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Calculate loss ONLY on the training edges
        loss = criterion(out[train_indices], data.y[train_indices].float())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")
            
    # 6. Evaluation Loop
    print("\nEvaluating GNN on Test Edges...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        
        test_logits = out[test_indices]
        test_probs = torch.sigmoid(test_logits)
        test_labels = data.y[test_indices]
        
        print("\n--- GNN Classification Report (Threshold: 0.50) ---")
        print(classification_report(test_labels.numpy(), (test_probs >= 0.50).long().numpy()))

        print("\n--- GNN Classification Report (Threshold: 0.95) ---")
        print(classification_report(test_labels.numpy(), (test_probs >= 0.95).long().numpy()))

        print("\n--- GNN Classification Report (Threshold: 0.99) ---")
        print(classification_report(test_labels.numpy(), (test_probs >= 0.99).long().numpy()))
        
    # 7. Save the trained model
    model_save_path = "results/fraud_gat.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel state dictionary saved to {model_save_path}")

if __name__ == "__main__":
    main()
