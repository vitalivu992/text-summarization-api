FROM python:3.8-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 5000/tcp
VOLUME ["/app/cnn_dailymail", "/app/pegasus-cnn_dailymail"]

COPY main.py inlinemodel.py nlpconnector.py ./

CMD [ "python3", "main.py"]