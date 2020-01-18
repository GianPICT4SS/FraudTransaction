import json
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


from utils.config import *
from kafka import KafkaProducer


producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER_URL)

def publish_prediction(pred):

    try:
        if pred.item() == 0:
            logger.info('PUBLISH PREDICTION')
            producer.send(LEGIT_TOPIC,
                          json.dumps({'STATUS': 'LEGIT', 'Prediction': float(pred.item())}).encode('utf-8'))
            #producer.flush()
        elif pred.item() == 1:
            logger.info('PUBLISH PREDICTION')
            producer.send(FRAUD_TOPIC,
                          json.dumps({'STATUS': 'FRAUD', 'Prediction': float(pred.item())}).encode('utf-8'))
            #producer.flush()

    except Exception as e:
        logger.error(e)



def publish_training_completed(model_id):

    try:
        producer.send(RETRAIN_TOPIC,
                  json.dumps({'training_completed': True, 'model_id': model_id}).encode('utf-8'))
        #producer.flush()
        logger.info('PUBLISH TRAINING COMPLETED')
    except Exception as e:
        logger.info(e)


def read_messages_count(path, repeat_every):
    file_list = list(path.iterdir())
    nfiles = len(file_list)
    if nfiles == 0:
        return 0
    else:
        return ((nfiles-1)*repeat_every) + len(file_list[-1].open().readlines())


def append_message(message, path, batch_id):
    logger.info('append msg')
    message_fname = f'messages_{batch_id}.csv'
    path_msg = path / message_fname
    try:
        with open(path_msg, "a") as f:
            f.write("%s\n" % (json.dumps(message)))
            logger.info('message saved on disk.')
    except Exception as e:
        logger.info(e)

    #df = pd.DataFrame(message)
    #df.to_csv(path_msg)




def send_retrain_message(model_id, batch_id):
    logger.info('Send retrain message')
    try:
        producer.send(RETRAIN_TOPIC,
                  json.dumps({'retrain': True, 'model_id': model_id, 'batch_id': batch_id}).encode('utf-8'))
        #producer.flush()
    except Exception as e:
        logger.info(e)