import os
import logging

# KAFKA BROKER
KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')


# TOPIC
TRANSACTIONS_TOPIC = os.environ.get('TRANSACTIONS_TOPIC')
RETRAIN_TOPIC = os.environ.get('RETRAIN_TOPIC')
LEGIT_TOPIC = os.environ.get('LEGIT_TOPIC')
FRAUD_TOPIC = os.environ.get('FRAUD_TOPIC')

# logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)
