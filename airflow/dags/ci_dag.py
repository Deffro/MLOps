import sys
sys.path.append("/opt/airflow/")
import os
from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.timezone import datetime

from src.processing.data_transformation import read_data, split_train_test, preprocess_data
from src.modeling.model_train import train_model
from src.modeling.model_push import register_model, register_model_by_comparison, check_if_registered_model_exists
from src.modeling.model_validation import evaluate_model
from src.config.core import DATA1


model_name = 'LR'
experiment_name = "churn_prediction"
reg_model = "LR/2"

_root_dir = "/"
_data_dir = "./data"
_output_data_dir = _data_dir + '/output_data'
if not os.path.exists(_output_data_dir):
    os.makedirs(_output_data_dir, mode=0o777)
_data_files = {
    'input_raw_data_file': os.path.join(DATA1),
    'raw_data_file': os.path.join(_output_data_dir + '/data.csv'),
    'raw_x_train_file': os.path.join(_output_data_dir + '/x_train_raw.csv'),
    'raw_x_test_file': os.path.join(_output_data_dir + '/x_test_raw.csv'),
    'raw_y_train_file': os.path.join(_output_data_dir + '/y_train_raw.csv'),
    'raw_y_test_file': os.path.join(_output_data_dir + '/y_test_raw.csv'),
    'transformed_x_train_file': os.path.join(_output_data_dir + '/x_train.csv'),
    'transformed_y_train_file': os.path.join(_output_data_dir + '/y_train.csv'),
    'transformed_x_test_file': os.path.join(_output_data_dir + '/x_test.csv'),
    'transformed_y_test_file': os.path.join(_output_data_dir + '/y_test.csv'),
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
    operator_read_data = PythonOperator(
        task_id='read_data',
        python_callable=read_data,
        op_kwargs={'data_files': _data_files}
    )

    operator_split_train_test = PythonOperator(
        task_id='split_train_test',
        python_callable=split_train_test,
        op_kwargs={'data_files': _data_files}
    )

    operator_preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        op_kwargs={'data_files': _data_files}
    )

    operator_model_training = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={
            'data_files': _data_files,
            'experiment_name': experiment_name,
            'model_name': model_name,
            'track_cv_performance': True
        }
    )

    operator_register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        op_kwargs={
            'model_name': model_name
        }
    )

    operator_check_if_registered_model_exists = BranchPythonOperator(
        task_id='check_if_registered_model_exists',
        python_callable=check_if_registered_model_exists,
        op_kwargs={
            'reg_model': reg_model
        }
    )

    operator_stop = DummyOperator(
        task_id='stop',
        dag=dag,
        trigger_rule="all_done",
    )

    operator_register_model_by_comparison = PythonOperator(
        task_id='register_model_by_comparison',
        python_callable=register_model_by_comparison,
        op_kwargs={
            'data_files': _data_files,
            'registered_model_uri': reg_model,
            'model_name': model_name,
            'push_to_production': True,
        },
    )

    operator_read_data >> operator_split_train_test >> operator_preprocess_data >> \
        operator_model_training >> operator_check_if_registered_model_exists >> \
        [operator_stop, operator_register_model, operator_register_model_by_comparison]
