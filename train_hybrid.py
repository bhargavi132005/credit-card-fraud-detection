import pandas as pd
import torch
import os
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gnn_model import FraudGAT
from src.preprocessing import load_data, preprocess_data
from src.features import create_features
from src.evaluate import evaluate_model, plot_roc_curve


def main():
    print("--- HYBRID MODEL PIPELINE ---")
    
    # 1. Load tabular data (Must be same size as graph data)
    df = load_data(nrows=1000000)
    df = preprocess_data(df)
    df = create_features(df)
    
    # 2. Load Graph Data & Trained GNN Model
    graph_path = "results/graph_data.pt"
    model_path = "results/fraud_gat.pth"
    
    if not os.path.exists(graph_path) or not os.path.exists(model_path):
        print("Missing graph data or GNN model! Run main.py and train_gnn.py first.")
        return
        
    print("Loading graph and trained GNN...")
    data = torch.load(graph_path, weights_only=False)
    
    node_in_dim = data.x.size(1)
    edge_in_dim = data.edge_attr.size(1)
    hidden_dim = 16
    
    gnn = FraudGAT(node_in_dim, edge_in_dim, hidden_dim)
    gnn.load_state_dict(torch.load(model_path, weights_only=True))
    gnn.eval()
    
    # 3. Extract GNN Embeddings
    print("Extracting network embeddings from GNN...")
    with torch.no_grad():
        embeddings = gnn(data.x, data.edge_index, data.edge_attr, extract_features=True)
        
    # 4. Merge Embeddings with Tabular Data
    print("Merging network features with tabular data...")
    embeddings_np = embeddings.numpy()
    for i in range(hidden_dim):
        df[f'gnn_emb_{i}'] = embeddings_np[:, i]
        
    # 5. Train Hybrid XGBoost
    print("Training Hybrid XGBoost...")
    X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    y = df['isFraud']

    # Define the train-test splits to evaluate
    splits = {
        "60:40": 0.4,
        "70:30": 0.3,
        "80:20": 0.2,
        "90:10": 0.1
    }
    
    report_lines = ["ML Model Name: Hybrid XGBoost (Tabular + GNN Embeddings)"]
    
    for split_name, test_size in splits.items():
        print(f"\n--- Training and Evaluating for split {split_name} ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Calculate scale_pos_weight to force the model to over-predict fraud
        neg_cases = (y_train == 0).sum()
        pos_cases = (y_train == 1).sum()
        scale_weight = neg_cases / pos_cases

        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', scale_pos_weight=scale_weight)
        xgb.fit(X_train, y_train)
        
        print(f"Evaluating Hybrid Model for {split_name}...")
        evaluate_model(xgb, X_test, y_test, threshold=0.02)
        
        y_probs = xgb.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= 0.02).astype(int)
        
        # Calculate metrics for the final summary report
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        sensitivity = recall
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
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
        report_lines.append(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

    print("\n\n" + "="*60)
    final_report = "\n".join(report_lines)
    print(final_report)
    print("="*60 + "\n")
    
    with open("results/split_performance_report.txt", "w") as f:
        f.write(final_report)

    # 7. Save the Hybrid Model
    print("\nSaving Hybrid Model...")
    hybrid_model_path = "results/xgboost_hybrid.joblib"
    joblib.dump(xgb, hybrid_model_path)
    print(f"Hybrid model saved to {hybrid_model_path}")

    # 8. Plot Feature Importances
    print("\nPlotting Feature Importances...")
    importances = xgb.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (Hybrid Model)')
    plt.tight_layout()
    importance_path = "results/hybrid_feature_importances.png"
    plt.savefig(importance_path)
    plt.close()
    print(f"Feature importances plot saved to {importance_path}")

    # 9. Plot ROC Curve for the final split
    print("\nPlotting ROC Curve...")
    roc_path = "results/hybrid_roc_curve.png"
    plot_roc_curve(y_test, y_probs, output_path=roc_path)
    print(f"ROC curve saved to {roc_path}")

if __name__ == "__main__":
    main()