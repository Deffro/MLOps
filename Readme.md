# MLOps Pipeline with <a href="https://mlflow.org/" target="_blank"><img alt="MLflow" src="https://img.shields.io/badge/-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" height="27"/></a> + <a href="https://www.postgresql.org/" target="_blank"><img alt="PostgreSQL" src="https://img.shields.io/badge/-PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white" height="27"/></a> + <a href="https://aws.amazon.com/s3/" target="_blank"><img alt="Amazon S3" src="https://img.shields.io/badge/-Amazon S3-569A31?style=flat-square&logo=amazons3&logoColor=white" height="27"/></a> + <a href="https://airflow.apache.org/" target="_blank"><img alt="Apache Airflow" src="https://img.shields.io/badge/-Apache Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white" height="27"/></a>

A complete Machine Learning lifecycle. The pipeline is as follows: 

`1. Read Data`➙`2. Split train-test`➙`3. Preprocess Data`➙`4. Train Model`➙<br>
&emsp; &emsp; &emsp; ➙ `5.1 Register Model`<br>
&emsp; &emsp; &emsp; ➙ `5.2 Update Registered Model`<br>

Telco Customer Churn dataset from <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank">Kaggle</a>.

## Tech Stack
<a href="https://mlflow.org/" target="_blank"><img alt="MLflow" src="https://img.shields.io/badge/-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" height="20"/></a>: For experiment tracking and model registration<br>
<a href="https://www.postgresql.org/" target="_blank"><img alt="PostgreSQL" src="https://img.shields.io/badge/-PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white" height="20"/></a>: Store the MLflow tracking<br>
<a href="https://aws.amazon.com/s3/" target="_blank"><img alt="Amazon S3" src="https://img.shields.io/badge/-Amazon S3-569A31?style=flat-square&logo=amazons3&logoColor=white" height="20"/></a>: Store the registered MLflow models and artifacts<br>
<a href="https://airflow.apache.org/" target="_blank"><img alt="Apache Airflow" src="https://img.shields.io/badge/-Apache Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white" height="20"/></a>: Orchestrate the MLOps pipeline<br>
<a href="https://scikit-learn.org/stable/index.html" target="_blank"><img alt="Scikit-learn" src="https://img.shields.io/badge/-Sklearn-fa9c3c?style=flat-square&logo=scikitlearn&logoColor=white" height="20"/></a>: Machine Learning<br>
<a href="https://jupyter.org/" target="_blank"><img alt="Jupyter" src="https://img.shields.io/badge/-Jupyter-eb6c2d?style=flat-square&logo=jupyter&logoColor=white" height="20"/></a>: R&D<br>
<a href="https://www.python.org/" target="_blank"><img alt="Python" src="https://img.shields.io/badge/-Python-4B8BBE?style=flat-square&logo=python&logoColor=white" height="20"/></a>
<a href="https://www.anaconda.com/" target="_blank"><img alt="Anaconda" src="https://img.shields.io/badge/-Anaconda-3EB049?style=flat-square&logo=anaconda&logoColor=white" height="20"/></a>
<a href="https://www.jetbrains.com/pycharm/" target="_blank"><img alt="PyCharm" src="https://img.shields.io/badge/-PyCharm-41c473?style=flat-square&logo=pycharm&logoColor=white" height="20"/></a>
<a href="https://www.docker.com/" target="_blank"><img alt="Docker" src="https://img.shields.io/badge/-Docker Compose-0db7ed?style=flat-square&logo=docker&logoColor=white" height="20"/></a>
<a href="https://git-scm.com/" target="_blank"><img alt="Git" src="https://img.shields.io/badge/-Git-F1502F?style=flat-square&logo=git&logoColor=white" height="20"/></a>

## How to reproduce

1. Have <a href="https://docs.docker.com/get-docker/" target="_blank">Docker</a> installed and running.

Make sure `docker-compose` is installed:
```commandline
pip install docker-compose
```

2. Clone the repository to your machine.
```commandline
git clone https://github.com/Deffro/MLOps.git
```

3. Rename `.env_sample` to `.env` and change the following variables:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION
   - AWS_BUCKET_NAME


4. Run the docker-compose file

```commandline
docker-compose up --build -d
```

## Urls to access

- <a href="http://localhost:8080" target="_blank">http://localhost:8080<a/> for `Airflow`. Use credentials: airflow/airflow
- <a href="http://localhost:5000" target="_blank">http://localhost:5000<a/> for `MLflow`.
- <a href="http://localhost:8893" target="_blank">http://localhost:8893<a/> for `Jupyter Lab`. Use token: mlops


## Cleanup
Run the following to stop all running docker containers through docker compose
```commandline
docker-compose stop
```

or run the following to stop and delete all running docker containers through docker
```commandline
docker stop $(docker ps -q)
docker rm $(docker ps -aq)
```

Finally, run the following to delete all (named) volumes
```commandline
docker volume rm $(docker volume ls -q)
```
