FROM tensorflow/tensorflow:2.15.0-gpu
LABEL maintainer "Joao Victor da Fonseca Pinto <jodafons@lps.ufrj.br>"
USER root

RUN apt-get update
RUN apt-get install -y git
RUN apt install -y python3-virtualenv
RUN pip install --upgrade pip