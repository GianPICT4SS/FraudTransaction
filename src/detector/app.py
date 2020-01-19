"""detector app: it must classify a transaction as fraud or normal. The prediction will be send by a producer
in the message_utils script."""

import json
import time
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


import pandas as pd


from utils.config import *
from utils.messages_utils import append_message, read_messages_count, publish_prediction
import torch
from utils.fraud_net import FraudNet
from kafka import KafkaConsumer



from model.trainer import Trainer

RETRAIN_EVERY = 250
EXTRA_MODELS_TO_KEEP = 1
TOPICS = [TRANSACTIONS_TOPIC, RETRAIN_TOPIC]

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


#logger.info('Loading the Pytorch Model...')
#try:
#    model = load_checkpoint(MODELS/'checkpoint_0.pth')
#except Exception as e:
#    logger.exception(str(e))

def is_retraining_message(msg):
    """This method allows the detector to know if a new model is available by reading a msg from a topic."""
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

    row = df.iloc[:, :-1].values  # don't take the target.
    row = torch.from_numpy(row).float()
    output = model(row)
    _, predicted = torch.max(output.data, 1)
    return predicted


def start(model_id, messages_count, batch_id):

    for msg in consumer:
        message = json.loads(msg.value)

        if is_retraining_message(msg):
            # A new model is available
            model_fname = f'model_{model_id}.pth'
            model = load_checkpoint(MODELS/model_fname)
            logger.info(f'New model reloaded {model_id}')

        elif is_application_message(msg):
            pred = predict(message)  # get the prediction
            publish_prediction(pred)  # publish prediction msg
            print(message)
            append_message(message)  #save the transaction in order to increase the train dataset
            messages_count += 1
            if messages_count % RETRAIN_EVERY == 0:
                # TO DO: here start the thread trainer, its run method starts the retrain
                # trainer(model_id, batch_id)
                model_id = model_id + 1
                logger.info(f'loading model {model_id}, and batch_id, messages_count = {batch_id, messages_count}')
                Trainer(model_id, batch_id)
                batch_id += 1






if __name__ == '__main__':



    messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
    batch_id = messages_count % RETRAIN_EVERY

    model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
    logger.info(f'loading model {model_id}, and batch_id, messages_count = {batch_id, messages_count}')
    model_fname = f'checkpoint_{model_id}.pth'
    model = load_checkpoint(MODELS / model_fname)
    
    time.sleep(5)
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER_URL)
    consumer.subscribe(TOPICS)

    start(model_id, messages_count, batch_id)




