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
    GENERATOR_start.mkdir(exist_ok=True)
    DETECTOR_start.mkdir(exist_ok=True)
    TRAIN_start.mkdir(exist_ok=True)
    TEST_start.mkdir(exist_ok=True)
    MODELS_start.mkdir(exist_ok=True)
    #DATA_PROCESSOR_start.mkdir(exist_ok=True)

def download_data(train_size=0.8):

    logger.info('Downloading data...')

    df = pd.read_csv('files/creditcard.csv')
    N_train = int(df.shape[0]*train_size)
    df_train = df.iloc[:N_train, :]
    df_test = df.iloc[N_train:, :]

    print('before')
    print(f'dfTrain_shape: {df_train.shape}, dfTest_shape:{df_test.shape}')

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    logger.info('Creating train dataset.')
    df_train.to_csv(str(TRAIN_start) + '/df_train.csv', index=False)
    df_test.to_csv(str(TRAIN_start) + '/df_test.csv', index=False)

    logger.info('Creating test dataset.')
    df_test.to_csv(str(TEST_start) + '/df_test.csv', index=False)


def create_data_processor():
    logger.info("creating preprocessor...")

    dataprocessor = build_train(str(TRAIN_start/'X_train.csv')) #str(DATA_PROCESSOR_start))
    return dataprocessor

if __name__ == '__main__':

    create_folder()
    df = download_data()
    #dataprocessor = create_data_processor()






