help: 
    echo "==== Help ==="
    echo "to create a network run make network"
    echo "to build the project use make build"
    echo "to up the project use make up"
    echo "to build and run the project use make build-run"
	echo "to stop the detector and the generator use make down"
	echo "to stop the kafka cluster use make down-kafka"
	echo "to delete the network use make delete-net"

network:
    docker network create kafka-net-fraud

build:
    docker-compose -f docker-compose.kafka.yml logs -f broker | grep "started"

up:
    docker-compose up -d

down:
    docker-compose down

down-kafka:
    docker-compose -f docker-compose.kafka.tml down

delete-net:
    docker network rm kafka-net-fraud

