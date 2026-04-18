from sklearn.metrics import classification_report


def evaluate_model(model, X_test, y_test, output_path=None):

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n")
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nClassification report saved to {output_path}")

    return y_pred