import pandas as pd


def load_data(path="data/paysim.csv", nrows=1000000):
    print("Loading dataset...")
    df = pd.read_csv(path, nrows=nrows)
    return df


def basic_info(df):
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFraud Distribution:")
    print(df['isFraud'].value_counts())


def preprocess_data(df):
    df = pd.get_dummies(df, columns=['type'], drop_first=True, dtype=int)
    return df