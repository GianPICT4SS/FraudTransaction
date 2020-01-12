"""Utilities for generating fakes transactions."""

import random

def create_random_transaction(df):
    """Create a fake, randomised transaction."""

    shape_x = df.shape[0]
    index = random.randint(0, shape_x-1)
    transaction = df.iloc[index:index+1, :]
    transaction = transaction.to_dict()
    return transaction
