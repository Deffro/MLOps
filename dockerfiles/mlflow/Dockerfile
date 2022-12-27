FROM python:3.9-slim-buster
RUN apt-get update
RUN apt-get install -y gcc g++ postgresql postgresql-contrib postgresql-server-dev-all python-psycopg2
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt