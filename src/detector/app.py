"""detector app: it must classify a transaction as fraud or normal."""

import json
import logging
import pickle

from utils.config import *
from utils.messages_utils import append_message, read_messages_count, send_retrain_message, publish_prediction
import torch
from model.model_1 import FraudNet
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


MESSAGES_PATH = DETECTOR/'messages'
RETRAIN_EVERY = 150
EXTRA_MODELS_TO_KEEP = 1
TOPICS = [TRANSACTIONS_TOPIC, RETRAIN_TOPIC]
column_order = pickle.load(open(DATA_PROCESSOR/'column_order.p', 'rb'))
dataprocessor = None
consumer = None
model = None

####################################
# Load Pytorch Model
####################################
if not torch.cuda.is_available():
    model = FraudNet()
    model = model.load_state_dict(torch.load(MODELS/'model_1.pt', map_location=torch.device('cpu')))
    model.eval()

else:
    model = FraudNet()
    model = model.load_state_dict(torch.load(MODELS/'model_1.pt'))
    model.eval()



def reload_model(path):
    """this method allows to reload a new model, when it is available (due to the retraining action)"""
    return pickle.load(open(path, 'rb'))

def is_retraining_message(msg):
    """This method allows the detector to know if a new model is available by reading a msg from a topic.
    N.B in our app all msg are already in json format"""
    message = json.loads(msg.value)
    return msg.topic == 'RETRAIN_TOPIC' and 'training_completed' in message and message['training_completed']

def is_application_message(msg):
    """This method is necessary for discerning if the incoming msg comes from transaction or not."""
    return msg.topic == 'TRANSACTION_TOPIC'


def predict(message):
    """classify the transaction by using the pre-trained model"""

    pass


def start(model_id, messages_count, batch_id):

    for msg in consumer:
        message = json.loads(msg.value)

        if is_retraining_message(msg):
            model_fname = f'model_{model_id}.pt'
            model = reload_model(MODELS/model_fname)
            logger.info(f'New model reloaded {model_id}')

        elif is_application_message(msg):
            request_id = message['request_id']
            #pred = predict(message['data'], column_order)
            #publish_prediction(pred, request_id)

            append_message(message['data'], MESSAGES_PATH, batch_id)
            messages_count += 1
            if messages_count % RETRAIN_EVERY == 0:
                model_id = (model_id + 1) % (EXTRA_MODELS_TO_KEEP + 1)
                send_retrain_message(model_id, batch_id)
                batch_id += 1




if __name__ == '__main__':

    dataprocessor_id = 0
    dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
    dataprocessor = pickle.load(open(DATA_PROCESSOR / dataprocessor_fname, 'rb'))

    messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
    batch_id = messages_count % RETRAIN_EVERY

    model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
    model_fname = 'model_{}_.p'.format(model_id)
    model = reload_model(MODELS / model_fname)

    consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER_URL)
    consumer.subscribe(TOPICS)

    start(model_id, messages_count, batch_id)
