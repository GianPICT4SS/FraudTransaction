"""This script initialise the system. It must split the dataset into an historical dataset to be used as train set
and a test dataset to be used as fake unseen transaction. It is thought as an initialization of the simulation.
This means it should be running just one time at the begin. This is useful since the huge size of the dataset, this
allows to reduce the computation time due to the future input/output operation"""

import argparse
import logging
import pandas as pd
from utils.preprocess_data import build_train
from utils.config import *
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Parser for the random_seed
parser = argparse.ArgumentParser(description='FraudTransaction')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--test_size', type=int, default=0.2,
                    help='test size')
args = parser.parse_args()


def create_folder():
    logger.info('Creating directory structure...')
    GENERATOR.mkdir(exist_ok=True)
    DETECTOR.mkdir(exist_ok=True)
    TRAIN.mkdir(exist_ok=True)
    TEST.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    DATA_PROCESSOR.mkdir(exist_ok=True)

def download_data():

    logger.info('Downloading data...')
    df = pd.read_csv('files/creditcard.csv')
    X = df.iloc[:, :-1]
    y = df.pop('Class')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)


    logger.info('Creating train dataset.')
    X_train.to_csv(str(TRAIN) + '/X_train.csv')
    y_train.to_csv(str(TRAIN) + '/y_train.csv', header=False)
    y_test.to_csv(str(TRAIN) + '/y_test.csv', header=False)

    logger.info('Creating test dataset.')
    X_test.to_csv(str(TEST) + '/X_test.csv')

def create_data_processor():
    logger.info("creating preprocessor...")
    dataprocessor = build_train(str(TRAIN/'X_train.csv'), str(DATA_PROCESSOR))
    return dataprocessor

if __name__ == '__main__':

    #create_folder()
    #download_data()
    dataprocessor = create_data_processor()






