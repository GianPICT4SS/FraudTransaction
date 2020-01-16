import os
from pathlib import Path





# Path directory to be created for the dataset
path_gen_start = Path('src/generator')
path_det_start = Path('src/detector')

path_det = Path('./')
path_gen = Path('./')

GENERATOR_start = path_gen_start/'dataset'
TEST_start = GENERATOR_start/'test'
DETECTOR_start = path_det_start/'dataset'
TRAIN_start = DETECTOR_start/'train'
MODELS_start = path_det_start/'model/models'
DATA_PROCESSOR_start = path_det_start/'dataset/dataprocessor'
MESSAGES_PATH_start = path_det_start/'dataset/messages'
MODELS = path_det/'model/models'
DATA_PROCESSOR = path_det/'dataset/dataprocessor'
MESSAGES_PATH = path_det/'dataset/messages'

# KAFKA BROKER
KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')


# TOPIC
TRANSACTIONS_TOPIC = os.environ.get('TRANSACTIONS_TOPIC')
RETRAIN_TOPIC = os.environ.get('RETRAIN_TOPIC')
LEGIT_TOPIC = os.environ.get('LEGIT_TOPIC')
FRAUD_TOPIC = os.environ.get('FRAUD_TOPIC')


