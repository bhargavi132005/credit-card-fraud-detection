import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
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
    
    # 2. Train/Test Split Iteration Setup
    num_edges = data.edge_index.size(1)
    indices = torch.randperm(num_edges)
    
    splits = {
        "60:40": 0.4,
        "70:30": 0.3,
        "80:20": 0.2,
        "90:10": 0.1
    }
    
    report_lines = ["ML Model Name: Graph Attention Network (FraudGAT)"]
    
    node_in_dim = data.x.size(1)
    edge_in_dim = data.edge_attr.size(1)
    hidden_dim = 16
    
    for split_name, test_size in splits.items():
        print(f"\n--- Training and Evaluating for split {split_name} ---")
        train_size = int((1 - test_size) * num_edges)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # 3. Handle Imbalance (Crucial for GNNs)
        num_neg = (data.y[train_indices] == 0).sum().item()
        num_pos = (data.y[train_indices] == 1).sum().item()
        
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float)
        print(f"Calculated Imbalance Weight (pos_weight): {pos_weight.item():.2f}")

        # 4. Initialize the Model, Optimizer, and Loss
        model = FraudGAT(node_in_dim, edge_in_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 5. Training Loop
        epochs = 100
        print("Starting GNN Training...")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_attr)
            
            # Calculate loss ONLY on the training edges
            loss = criterion(out[train_indices], data.y[train_indices].float())
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")
                
        # 6. Evaluation Loop
        print(f"Evaluating GNN on Test Edges for {split_name}...")
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            
            test_logits = out[test_indices]
            test_probs = torch.sigmoid(test_logits)
            test_labels = data.y[test_indices]
            
            y_pred = (test_probs >= 0.50).long().numpy()
            y_true = test_labels.numpy()
            
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            sensitivity = recall
            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, test_probs.numpy())
            error_rate = 1 - accuracy
            
            report_lines.append(f"\n--- {split_name} Split Results ---")
            report_lines.append(f"1. Confusion Matrix:\n{cm}")
            report_lines.append(f"2. F1-Score:    {f1:.4f}")
            report_lines.append(f"3. Precision:   {precision:.4f}")
            report_lines.append(f"4. Recall:      {recall:.4f}")
            report_lines.append(f"5. Sensitivity: {sensitivity:.4f}")
            report_lines.append(f"6. Accuracy:    {accuracy:.4f}")
            report_lines.append(f"7. AUC:         {auc:.4f}")
            report_lines.append(f"8. Error Rate:  {error_rate:.4f}\n")
            report_lines.append(f"Classification Report:\n{classification_report(y_true, y_pred)}\n")

    print("\n\n" + "="*60)
    final_report = "\n".join(report_lines)
    print(final_report)
    print("="*60 + "\n")
    
    with open("results/gnn_split_performance_report.txt", "w") as f:
        f.write(final_report)
        
    # 7. Save the trained model (saves the last split, 90:10)
    model_save_path = "results/fraud_gat.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel state dictionary saved to {model_save_path}")

if __name__ == "__main__":
    main()
