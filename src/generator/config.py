import os


# KAFKA BROKER
KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')


# TOPIC
TRANSACTIONS_TOPIC = os.environ.get('TRANSACTIONS_TOPIC')
RETRAIN_TOPIC = os.environ.get('RETRAIN_TOPIC')
LEGIT_TOPIC = os.environ.get('LEGIT_TOPIC')
FRAUD_TOPIC = os.environ.get('FRAUD_TOPIC')

