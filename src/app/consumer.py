""" It acts as a Kafka consumer for retriving information from the broker and making it available for the service_layer. """

import kafka import KafkaConsumer

from detector.utils.config import *

TOPICS = [TRANSACTIONS_TOPIC, FRAUD_TOPIC]