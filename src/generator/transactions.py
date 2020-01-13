"""Utilities for generating fakes transactions."""

import random

def create_random_transaction(df):
    """Create a fake, randomised transaction."""

    shape_x = df.shape[0]
    index = random.randint(0, shape_x-1)
    transaction = df.iloc[index:index+1, :]
    transaction['json'] = transaction.apply(lambda x: x.to_json(), axis=1)
    msg = transaction.json.to_list()
    #transaction = transaction.to_dict()
    return msg


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('dataset/X_test.csv')
    tr = create_random_transaction(df)