import json
import pickle
import pandas as pd

from utils.config import *
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER_URL)

def publish_prediction(pred):

    try:
        if pred.item() == 0:
            producer.send(LEGIT_TOPIC,
                          json.dumps({'STATUS': 'LEGIT', 'Prediction': float(pred.item())}).encode('utf-8'))
            producer.flush()
        elif pred.item() == 1:
            producer.send(FRAUD_TOPIC,
                          json.dumps({'STATUS': 'FRAUD', 'Prediction': float(pred.item())}).encode('utf-8'))
            producer.flush()

    except Exception as e:
        print(e)



def publish_traininig_completed(model_id):
    producer.send(RETRAIN_TOPIC,
                  json.dumps({'training_completed': True, 'model_id': model_id}).encode('utf-8'))
    producer.flush()


def read_messages_count(path, repeat_every):
    file_list = list(path.iterdir())
    nfiles = len(file_list)
    if nfiles == 0:
        return 0
    else:
        return ((nfiles-1)*repeat_every) + len(file_list[-1].open().readlines())


def append_message(message, path, batch_id):
    message_fname = f'messages_{batch_id}_.csv'
    df = pd.DataFrame(message)
    path_msg = path/message_fname
    df.to_csv(path_msg)




def send_retrain_message(model_id, batch_id):
    producer.send(RETRAIN_TOPIC,
                  json.dumps({'retrain': True, 'model_id': model_id, 'batch_id': batch_id}).encode('utf-8'))
    producer.flush()