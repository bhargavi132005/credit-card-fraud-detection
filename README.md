# ⚡ Neural Fraud Intelligence Engine (GraphBoost)

A real-time, graph-powered financial threat detection system. This project models financial transactions as a network graph and utilizes a cutting-edge hybrid architecture combining **Graph Attention Networks (GAT)** with **XGBoost** to detect anomalous and fraudulent behavior in highly imbalanced datasets.

## 🧠 Hybrid Architecture
Traditional fraud detection relies heavily on tabular features (e.g., transaction amounts, balances). This engine enhances traditional machine learning by injecting spatial/network context. 
1. **Graph Attention Network (FraudGAT):** Learns complex behavioral patterns by observing how money flows between different accounts (nodes) over edges (transactions).
2. **Hybrid XGBoost Ensemble:** Takes the 16-dimensional embeddings extracted by the GNN and merges them with standard tabular features, allowing the model to make predictions based on both *amount-based* anomalies and *network-based* anomalies (like money mule rings).

## ✨ Key Features
- **Dynamic Evaluation:** Automatically trains and evaluates the model across multiple split ratios (60:40 up to 90:10).
- **Interactive Dashboard:** Built with Streamlit, featuring a dark-themed, cyberpunk-inspired UI.
- **Real-Time Predictor:** Enter transaction details to get live fraud probability scoring. Features hardcoded fallback rules to catch impossible mathematical anomalies and classic account-draining signatures.
- **Graph Intelligence Visualization:** Dynamically renders transaction networks using Plotly to visually explain *why* a transaction was flagged.

## 📂 Project Structure
* `main.py`: Preprocesses the PaySim dataset, trains the baseline XGBoost model, and constructs the PyTorch Geometric graph data.
* `train_gnn.py`: Trains the Graph Attention Network on the graph data and saves the `.pth` model.
* `train_hybrid.py`: Extracts embeddings from the trained GAT, merges them with tabular data, and trains the final Hybrid XGBoost model.
* `app.py`: The Streamlit application that serves the interactive dashboard and predictions.
* `src/`: Core logic including feature engineering, model definitions, and evaluation metrics.
* `results/`: Contains trained `.joblib` / `.pth` models, evaluation reports, and metric charts.

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost torch torch_geometric networkx plotly streamlit pillow
```

### 2. Dataset
Download the **PaySim** dataset from Kaggle and place it in the `data/` directory:
* `data/paysim.csv`

### 3. Run the Pipeline
Execute the scripts in the following order to build the models:
```bash
# 1. Prepare data, train baseline, and build graph
python main.py

# 2. Train the Graph Attention Network
python train_gnn.py

# 3. Train the final Hybrid Model & generate charts
python train_hybrid.py
```

### 4. Launch the Dashboard
Start the Neural Fraud Intelligence Engine UI:
```bash
streamlit run app.py
```

---
*Built as a conceptual architecture for next-generation fintech security.*
