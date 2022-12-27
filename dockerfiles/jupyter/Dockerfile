FROM jupyter/scipy-notebook:lab-3.3.2
USER root
RUN apt-get update
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt