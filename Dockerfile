FROM python:3.6

LABEL maintainer="chris_thompson@manceps.com"

WORKDIR /opt

COPY requirements.txt .

RUN apt-get update && \
    python3 -m pip install --upgrade pip && \    
    pip3 install -r requirements.txt



COPY app/ .

CMD python3 app/routes.py


