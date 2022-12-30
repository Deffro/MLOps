FROM python:3.7-slim-buster
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all python-psycopg2
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt