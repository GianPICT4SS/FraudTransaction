# Microservice FraudTransaction App:
A real-time fraud transaction prediction model by using Kafka/Spark/Pytorch

The system simulate a real world application by using two microservice isolated into two docker-compose configuration.

The first one is the Kafka cluster, where the broker is always ready to dispatch message beyond clients want to use it or not. 
The second one is a customer application, where two services works. In order to be used togheter lcoally, a docker network is created and shared between them.

The customer application is composed by two services. The first one works as a generator transaction simulator. It simulate clients that generate transaction around the world, so it is just a kafka producer. The other one is a detector, it must classify the transaction and tells its prediction so it works as either a kafka consumer that a kafka producer. The detetector exploits a prediction model pre-trained with historically label data. It is composed by a deep-neural network (composed by linear hidden layer) built by using pytorch. 

The data preprocessing will be done by using Spark. 
