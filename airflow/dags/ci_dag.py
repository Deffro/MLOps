from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.timezone import datetime

from src.processing.data_transformation import read_data, split_train_test, preprocess_data
from src.modeling.model_train import train_model
from src.modeling.model_push import register_model, register_model_by_comparison, promote_model_to_production
from src.modeling.model_validation import evaluate_model


# SET A UNIQUE MODEL NAME (e.g. "model_<YOUR NAME>"):
_model_name = "my_model"
# SET A UNIQUE EXPERIMENT NAME (e.g. "experiment_<YOUR NAME>"):
_mlflow_experiment_name = "my_experiment"

_raw_data_dir = '/data/batch1'
# _raw_data_dir = '/data/batch2'


_root_dir = "/"
_data_dir = "/data"
_data_files = {
    'raw_data_file': os.path.join(_data_dir, 'data.csv'),
    'raw_train_file': os.path.join(_data_dir, 'data_train.csv'),
    'raw_test_file': os.path.join(_data_dir, 'data_test.csv'),
    'transformed_x_train_file': os.path.join(_data_dir, 'x_train.csv'),
    'transformed_y_train_file': os.path.join(_data_dir, 'y_train.csv'),
    'transformed_x_test_file': os.path.join(_data_dir, 'x_test.csv'),
    'transformed_y_test_file': os.path.join(_data_dir, 'y_test.csv'),
}

if not _root_dir:
    raise ValueError('PROJECT_PATH environment variable not set')

default_args = {
    'owner': 'Deffro',
    'depends_on_past': False,
    'start_date': days_ago(0),
    'retries': 1,
    'retry_delay': timedelta(seconds=5)
}

dag = DAG(
    'ci_pipeline',
    default_args=default_args,
    description='Continuous Integration Pipeline',
    schedule_interval=timedelta(days=1),
)

with dag:
    pass
    read_data = PythonOperator(
        task_id='read_data',
        python_callable=read_data,
        op_kwargs={}
    )

    split_train_test = PythonOperator(
        task_id='split_train_test',
        python_callable=split_train_test,
        op_kwargs={'data': _data_files,
                   'n_days_test': 20}
    )

    data_validation = PythonOperator(
        task_id='data_validation',
        python_callable=validate_data,
        op_kwargs={'data_files': _data_files,
                   'configs_dir': _data_dir}
    )

    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=transform_data,
        op_kwargs={'data_files': _data_files}
    )

    model_training = PythonOperator(
        task_id='model_training',
        python_callable=train_model,
        op_kwargs={
            'data_files': _data_files,
            'experiment_name': _mlflow_experiment_name
        }
    )

    model_validation = BranchPythonOperator(
        task_id='model_validation',
        python_callable=validate_model,
        op_kwargs={
            'data_files': _data_files,
            'model': _model_name
        },
    )

    stop = DummyOperator(
        task_id='keep_old_model',
        dag=dag,
        trigger_rule="all_done",
    )

    push_to_production = PythonOperator(
        task_id='push_new_model',
        python_callable=push_model,
        op_kwargs={
            'model': _model_name
        },
    )

    data_ingestion >> data_split >> data_validation >> data_transformation >> model_training >> model_validation >> [
        push_to_production, stop]
    data_split >> data_validation >> data_transformation >> model_validation >> [
        push_to_production, stop]