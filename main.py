from src.preprocessing import load_data, basic_info, preprocess_data
from src.features import create_features
from src.model_baseline import train_baseline
from src.evaluate import evaluate_model
from src.graph_construction import create_graph_data
import torch
import os
import joblib


def main():

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data(nrows=15000)

    # Info
    basic_info(df)

    # Preprocess
    df = preprocess_data(df)

    # Feature engineering
    df = create_features(df)

    # Train baseline
    model, X_test, y_test = train_baseline(df)

    # Save model
    model_path = os.path.join(output_dir, "xgboost_baseline.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Evaluate
    report_path = os.path.join(output_dir, "baseline_report.txt")
    evaluate_model(model, X_test, y_test, output_path=report_path)

    # --- DAY 2: Graph Construction ---
    graph_data, _ = create_graph_data(df)

    # Save graph data object
    graph_path = os.path.join(output_dir, "graph_data.pt")
    torch.save(graph_data, graph_path)
    print(f"\nGraph data object saved to {graph_path}")


if __name__ == "__main__":
    main()