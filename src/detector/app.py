"""detector app: it must classify a transaction as fraud or normal."""

import os
import json
import logging
import pickle

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

print(f'Current Directory: {os.getcwd()}')
MESSAGES_PATH = 'detector/messages'
RETRAIN_EVERY = 150
EXTRA_MODELS_TO_KEEP = 1
TOPICS = [TRANSACTIONS_TOPIC, RETRAIN_TOPIC]
#path_co = str(DATA_PROCESSOR) + '/column_order.p'
#with open(path_co, 'rb') as f:
#    column_order = pickle.load(f)
#column_order = sc.pickleFile('dataset/dataprocessor/column_order.p')


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
model = load_checkpoint(MODELS/'checkpoint_1.pth')



"""

if not torch.cuda.is_available():

    model = load_checkpoint(MODELS/'checkpoint_1.pth')
    device = torch.device('cpu')
    model.to(device)
    #model = model.load_state_dict(torch.load(MODELS/'model_1.pt', map_location=torch.device('cpu')))
    #state_dict = torch.load(MODELS/ 'model_1.pth', map_location=device)
    #print(f'TYPE: {type(state_dict)}')
    #for key, values in state_dict.items():
    #    print(key, values)
    #model = model.load_state_dict(state_dict)
    #model.eval()
    #try:
    #    logger.info('model.eval() OK')
    #    model.eval()
    #except Exception as e:
    #    logger.info(f'{e}')
    #    pass
else:
    device = torch.device('cuda')
    state_dict = torch.load(MODELS / 'model_1.pth')
    model = model.load_state_dict(torch.load(state_dict))
    model.to(device)
    try:
        model.eval()
    except:
        pass
"""




def reload_model(path):
    """this method allows to reload a new model, when it is available (due to the retraining action)"""
    #return sc.pickleFile(path)
    with open(str(path), 'rb') as f:
        return pickle.loads(f)

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

    """
    dataprocessor_id = 0
    dataprocessor_fname = f'dataprocessor_{dataprocessor_id}_.p'
    path_dp = str(DATA_PROCESSOR) + '/' + dataprocessor_fname
    with open(path_dp, 'rb') as f:
        dataprocessor = pickle.load(f)
    #dataprocessor = sc.pickleFile(dataprocessor_fname)

    messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
    batch_id = messages_count % RETRAIN_EVERY

    model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
    model_fname = 'model_{}_.p'.format(model_id)
    model = reload_model(MODELS / model_fname)
    
    """
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER_URL)
    consumer.subscribe(TOPICS)

    #start(model_id, messages_count, batch_id)
    import pandas as pd
    import numpy as np

    for msg in consumer:
        msg = msg.value
        print(f'Before msg_type {type(msg)}, len: {len(msg)}, type_ : {type(msg[0])}')
        msg = json.loads(msg) # List[0] = string
        print(f'After msg_type {type(msg)}, len: {len(msg)}, type_ : {type(msg[0])}')
        msg = json.dumps(msg[0])
        msg = json.loads(msg)  # ancora una stringa
        print(f'After msg_type {type(msg)}, len: {len(msg)}, type_ : {type(msg[0])}')




        print(f'value: {msg}, type: {type(msg)}')  # Class str


        #print(f'dirct: {dict_row}, type:{type(dict_row)}')
        row = np.array(([list(item.values()) for item in msg.values()]))
        #df = pd.DataFrame(msg)
        #row = np.asarray(df.values)
        print(f'ROW: {row}, type: {type(row)}')
        row = row.reshape(row.shape[0], 1)
        row = torch.from_numpy(row).double()
        # Try prediction
        output = model(row)
        _, predicted = torch.max(output.data, 1)
        print('#####################################################')
        print(f'prediction:{predicted}')
