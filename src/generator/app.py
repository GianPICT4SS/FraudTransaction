"""Produce fake transactions into a Kafka topic."""

import os
from time import sleep
import json
import pandas as pd
import threading

from kafka import KafkaProducer, KafkaConsumer
from transactions import create_random_transaction
from config import *


TRANSACTIONS_PER_SECOND = float(os.environ.get('TRANSACTIONS_PER_SECOND'))
SLEEP_TIME = 1 / TRANSACTIONS_PER_SECOND

TOPICS = [FRAUD_TOPIC, LEGIT_TOPIC]

###########################################
# Load Test Data
###########################################
df_test = pd.read_csv('dataset/test/df_test.csv')
X_test = df_test.iloc[:, :-1]


def start_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER_URL
        # Encode all values as JSON
        # value_serializer=lambda value: json.dumps(value).encode()  # produce json msg

    )
    while True:
        transaction = create_random_transaction(X_test)
        producer.send(TRANSACTIONS_TOPIC, value=json.dumps(transaction).encode('utf-8'))
        producer.flush()
        print(transaction)  # DEBUG
        # sleep(SLEEP_TIME)
        sleep(5)

def start_consumer():
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER_URL)
    consumer.subscribe(TOPICS)

    for msg in consumer:
        message = json.loads(msg.value)
        if "Prediction" in message:
            print(f"** CONSUMER: Received prediction {message['Prediction']}")
            print(f"Type Transaction: {message['STATUS']}")


#threads = []
t1 = threading.Thread(target=start_consumer)
t2 = threading.Thread(target=start_producer)
t1.setDaemon(True)
t2.setDaemon(True)
#threads.append(t1)
#threads.append(t2)
t1.start()
t2.start()


