from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def train_baseline(df):

    X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    return model, X_test, y_test