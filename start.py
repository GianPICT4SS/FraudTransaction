"""This script initialise the system. It must split the dataset into an historical dataset to be used as train set
and a test dataset to be used as fake unseen transaction. It is thought as an initialization of the simulation.
This means it should be running just one time at the begin. This is useful since the huge size of the dataset, this
allows to reduce the computation time due to the future input/output operation"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Parser for the random_seed
parser = argparse.ArgumentParser(description='FraudTransaction')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--test_size', type=int, default=0.2,
                    help='test size')
args = parser.parse_args()

# Path directory to be created for the dataset

path_gen = Path('src/generator')
path_det = Path('src/detector')

gen = path_gen/'dataset'
det = path_det/'dataset'

gen.mkdir(exist_ok=True)
det.mkdir(exist_ok=True)

########################################################
# Load Data
########################################################
df = pd.read_csv('files/dataset/creditcard.csv')
X = df.iloc[:, :-1]
y = df.pop('Class')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

X_train.to_csv(str(det) + '/X_train.csv')
y_train.to_csv(str(det) + '/y_train.csv', header=False)
y_test.to_csv(str(det) + '/y_test.csv', header=False)

X_test.to_csv(str(gen) + '/X_test.csv')





