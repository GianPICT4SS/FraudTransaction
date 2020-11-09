# FraudTransaction App:
A fully containerised real-time fraud transaction prediction model by using Kafka/Spark/Pytorch.

The system simulate a real world application by using two microservices isolated into two docker-compose configurations.

The first one is the Kafka cluster, where the broker is always ready to dispatch message beyond clients want to use it or not. 
The second one is a customer application, where two services work. In order to be both used locally, a docker network is created and shared between them.

The customer application is composed by two services (microservices). The first one works as a generator transaction simulator. It simulate clients that generate transaction around the world, and it is also a consumer because it simulates a client that wants to know if its transactions are fraud or not. The other one is a detector, it must classify the transactions and tells its prediction so it works as either a kafka consumer and a kafka producer. The detetector exploits a prediction model pre-trained with historically label data. It is composed by a deep-neural network (composed by linear hidden layer) built by using pytorch. The detector is able to re-train the model in real-time, without stopping the prediction service. Indeed, when a number of transactions are done the train dataset is refresh with new data and label, in order to train a new model. When a new model is available, the detector loads and uses it for the prediction phase.   

The data preprocessing is done by using Spark and is implemented on the detector microservice. 

**Install**
This fraud transaction system is fully containerised. You will need Docker and Docker Compose for running it.

Below the sequentially steps needed to run the system.

Locally: 
1) run the start.py script (this will create all the system directory and the splits the dataset.)

Docker steps:

0) Create a Docker network: 
$ docker network create kafka-net-fraud

**Quickstart**
1) Spin up the local single-node Kafka cluster:
$ docker-compose -f docker-compose.kafka.yml up -d
2) Check the cluster is up and running:
$ docker-compose -f docker-compose.kafka.yml logs -f broker | grep "started"
3) Start the transaction generator and the fraud detector:
$ docker-compose up -d

**Shut down**
1) To stop the transaction generator and the fraud detector:
$ docker-compose down
2) To stop the Kafka cluster:
$ docker-compose -f docker-compose.kafka.tml down
3) To remove the Docker network (if you execute this step you will need to recreate the network):
$ docker network rm kafka-net-fraud

N.B: The system has been tested by using a free dataset available on the url: https://www.kaggle.com/mlg-ulb/creditcardfraud
To run it, you should download this dataset, or use a your personal dataset.

Enjoy it and feel free to share/improve it!  
