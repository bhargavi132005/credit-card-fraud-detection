from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

def train_baseline(df):

    X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    y = df['isFraud']

    splits = {
        "60:40": 0.4,
        "70:30": 0.3,
        "80:20": 0.2,
        "90:10": 0.1
    }
    
    report_lines = ["ML Model Name: Baseline XGBoost"]
    
    for split_name, test_size in splits.items():
        print(f"\n--- Training Baseline for split {split_name} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Calculate scale_pos_weight to handle class imbalance
        neg_cases = (y_train == 0).sum()
        pos_cases = (y_train == 1).sum()
        scale_weight = neg_cases / max(1, pos_cases)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            scale_pos_weight=scale_weight
        )

        model.fit(X_train, y_train)
        
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= 0.05).astype(int)  # 0.05 threshold as used in main.py
        
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

    final_report = "\n".join(report_lines)
    os.makedirs("results", exist_ok=True)
    with open("results/xgboost_split_performance_report.txt", "w") as f:
        f.write(final_report)

    return model, X_test, y_test