""" It is a :
1) a thread in communication with detector that starts when a new dataset is available
2) Producer on RETRAIN_TOPIC.
It must do: a) see if a new train dataset is available, b) trains a new model and save it, c) alert the detector about
the possibility to use a new model.
"""


import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.utils_train import train
from utils.fraud_net import FraudNet
from utils.config import TRAIN, MESSAGES_PATH



from utils.preprocess_data import build_train

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudNet().float().to(device)


from threading import Thread, Event


class Trainer(Thread):
    

    def __init__(self, model_id, batch_id):
        Thread.__init__(self)
        self.model_id_ = model_id
        self.batch_id_ = batch_id

        self.stop_event = Event()
        self.start()


    def run(self):
        #Method called when the thread start...
        logger.info(f'Retraining for model_id: {self.model_id_}, started with {self.batch_id_}.')
        message_name = f'messages.txt'
        new_train_path = str(MESSAGES_PATH) + '/' + message_name
        print(f'new_train_path {new_train_path}')
        train_path = str(TRAIN) + '/df_train.csv'
        dtrain = build_train(train_path=train_path, new_train_path=new_train_path)  # return FeatureTools object
        train_rdd = dtrain.data.select('scaledFeatures')  # data already preprocessed
        y_rdd = dtrain.target
        train(model, train_rdd=train_rdd, y_rdd=y_rdd, model_id=self.model_id_, device=device)  # train and save checkpoint

