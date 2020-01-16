"""detector app: it must classify a transaction as fraud or normal."""

import os
import json
import logging
import pickle
import pandas as pd

from utils.config import *
from utils.messages_utils import append_message, read_messages_count, send_retrain_message, publish_prediction
import torch
from model.model_1 import FraudNet
from kafka import KafkaConsumer, KafkaProducer

import pyspark
from pyspark.sql.session import SparkSession
###################################################
# Spark Configuration
###################################################

#sc = pyspark.SparkContext(master='local', appName='FraudTransaction')
#spark = SparkSession(sc)


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


RETRAIN_EVERY = 150
EXTRA_MODELS_TO_KEEP = 1
TOPICS = [TRANSACTIONS_TOPIC, RETRAIN_TOPIC]

#dataprocessor = None
consumer = None
model = None

####################################
# Load Pytorch Model
####################################

model = FraudNet()

def load_checkpoint(filepath, device='cpu'):
    if device == 'cpu':
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model
    elif device == 'gpu':
        device = torch.device('cuda')
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(device)
        model.eval()
        return model


logger.info('Loading the Pytorch Model...')
model = load_checkpoint(MODELS/'checkpoint_0.pth')

def is_retraining_message(msg):
    """This method allows the detector to know if a new model is available by reading a msg from a topic.
    N.B in our app all msg are already in json format"""
    message = json.loads(msg.value)
    return msg.topic == RETRAIN_TOPIC and 'training_completed' in message and message['training_completed']

def is_application_message(msg):
    """This method is necessary for discerning if the incoming msg comes from transaction or not."""
    message = json.loads(msg.value)
    return msg.topic == TRANSACTIONS_TOPIC and 'prediction' not in message


def predict(message):
    """classify the transaction by using the pre-trained model
    TO DO: make the preprocessor of row"""

    df = pd.DataFrame(message)
    row = df.values
    row = torch.from_numpy(row).float()
    output = model(row)
    _, predicted = torch.max(output.data, 1)
    return predicted







def start(model_id, messages_count, batch_id):

    for msg in consumer:
        message = json.loads(msg.value)

        if is_retraining_message(msg):
            model_fname = f'model_{model_id}.pt'
            #model = load_checkpoint(MODELS/model_fname)
            logger.info(f'New model reloaded {model_id}')

        elif is_application_message(msg):
            pred = predict(message)
            publish_prediction(pred)

            append_message(msg, MESSAGES_PATH, batch_id)
            messages_count += 1
            if messages_count % RETRAIN_EVERY == 0:
                model_id = (model_id + 1) % (EXTRA_MODELS_TO_KEEP + 1)
                send_retrain_message(model_id, batch_id)
                batch_id += 1




if __name__ == '__main__':


    #dataprocessor_id = 0
    #dataprocessor_fname = f'dataprocessor_{dataprocessor_id}_.p'
    #path_dp = str(DATA_PROCESSOR) + '/' + dataprocessor_fname
    #with open(path_dp, 'rb') as f:
    #    dataprocessor = pickle.load(f)
    #dataprocessor = sc.pickleFile(dataprocessor_fname)

    messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
    batch_id = messages_count % RETRAIN_EVERY

    model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
    model_fname = f'checkpoint_{model_id}.pth'
    model = load_checkpoint(MODELS / model_fname)
    

    consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER_URL)
    consumer.subscribe(TOPICS)

    start(model_id, messages_count, batch_id)




