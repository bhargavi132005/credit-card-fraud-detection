def create_features(df):

    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    return df