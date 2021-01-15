FROM python:3.7-slim-buster

WORKDIR usr/FraudApp

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt
COPY ./ ./

CMD ["python", "app.py"]
