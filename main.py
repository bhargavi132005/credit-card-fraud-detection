from src.preprocessing import load_data, basic_info, preprocess_data
from src.features import create_features
from src.model_baseline import train_baseline
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from src.graph_construction import create_graph_data
import torch
import os
import joblib


def main():

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    # Load exactly 1 million rows to prevent memory crashes
    df = load_data(nrows=1000000)

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
    # Lowering the threshold intentionally drops accuracy to ~95-98% but catches much more fraud!
    y_pred = evaluate_model(model, X_test, y_test, output_path=report_path, threshold=0.05)

    # Visualizations
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, output_path=cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]

        roc_path = os.path.join(output_dir, "roc_curve.png")
        plot_roc_curve(y_test, y_probs, output_path=roc_path)
        print(f"ROC curve saved to {roc_path}")

        pr_path = os.path.join(output_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(y_test, y_probs, output_path=pr_path)
        print(f"Precision-Recall curve saved to {pr_path}")

    # --- DAY 2: Graph Construction ---
    graph_data, _ = create_graph_data(df)

    # Save graph data object
    graph_path = os.path.join(output_dir, "graph_data.pt")
    torch.save(graph_data, graph_path)
    print(f"\nGraph data object saved to {graph_path}")


if __name__ == "__main__":
    main()