""" It is a :
1) Consumer on RETRAIN_TOPIC
2) Producer on RETRAIN_TOPIC.
It must do: a) see if a new train dataset is available, b) trains a new model and save it, c) alert the detector about
the possibility to use a new model.
"""

import json


from .fraud_net import FraudNet
from ..utils.feature_tools import FeatureTools
from ..utils.config import RETRAIN_TOPIC, KAFKA_BROKER_URL

import torch
import torch.nn as nn
import torch.utils.data as data_utils

from kafka import KafkaProducer, KafkaConsumer




def is_new_dataset_available():
    try:
        consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER_URL)
        consumer.subscribe(RETRAIN_TOPIC)
        for msg in consumer:
            message = json.loads(msg.value)

    except:
        pass





def train_new_model():
    pass

def send_alert():
    pass
