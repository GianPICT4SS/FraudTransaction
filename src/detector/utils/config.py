import os
from pathlib import Path





# Path directory to be created for the dataset
path_gen = Path('src/generator')
path_det = Path('src/detector')

GENERATOR = path_gen/'dataset'
TEST = GENERATOR/'test'
DETECTOR = path_det/'dataset'
TRAIN = DETECTOR/'train'
MODELS = path_det/'model/models'
DATA_PROCESSOR = DETECTOR/'dataprocessor'

# Global Variables
KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')
TRANSACTIONS_TOPIC = os.environ.get('TRANSACTIONS_TOPIC')
RETRAIN_TOPIC = os.environ.get('RETRAIN_TOPIC')
LEGIT_TOPIC = os.environ.get('LEGIT_TOPIC')
FRAUD_TOPIC = os.environ.get('FRAUD_TOPIC')
