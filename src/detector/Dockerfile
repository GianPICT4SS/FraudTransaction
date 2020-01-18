FROM python:3.7-slim-stretch

WORKDIR usr/FraudApp

RUN mkdir -p /usr/share/man/man1/
# Install OpenJDK-8 (useful for pyspark)
RUN apt-get update && \
apt-get install -y openjdk-8-jdk && \
apt-get install -y ant && \
apt-get clean;
# Fix certificate issues
RUN apt-get update && \
apt-get install ca-certificates-java && \
apt-get clean && \
update-ca-certificates -f;
# Setup JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/


COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt
COPY ./ ./
RUN chmod -R 777 ./dataset/messages


CMD ["python", "app.py"]
