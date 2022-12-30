import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from src.processing.data_transformation import check_keys
from loguru import logger


def get_val_performance(y_true: np.array, y_pred: np.array):
    """
    Get performance metrics for the validation set
    """
    val_accuracy = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average='macro')

    return val_accuracy, val_f1


def get_cv_performance(x_train: pd.DataFrame, y_train: np.array, model):
    """
    Get performance metrics for the train set using cross validation
    """
    # For CV model should be sklearn and not a loaded trained mlflow model
    if "sklearn" not in str(type(model)):
        raise TypeError("Model should be and sklearn model.")
    cv_accuracy = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_macro').mean()

    return cv_accuracy, cv_f1


def get_model_from_uri(model_uri):
    """
    Load a model that is either logged or registered on MLflow
    """
    model = None
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except:
        logger.error(f"There is no model {model_uri}.")

    return model


def get_predictions(x_test: pd.DataFrame, model):
    """
    Use a model to predict on given data
    """
    y_pred = model.predict(x_test)

    return y_pred


def evaluate_model(data_files, model_uri_=None, task_instance=None):
    """
    Pipeline to use the evaluation functions
    data_files (dict): A dictionary of data file paths.
    The keys that this function will use are:
        'transformed_x_test_file': the transformed x_test
        'transformed_y_test_file': the transformed y_test
    model_uri_: Pass a mlflow uri if running without using airflow
    When using airflow, this uri is passed through task_instance.xcom_pull(task_ids='train_model')
    """
    required_keys = [
        'transformed_x_test_file',
        'transformed_y_test_file'
    ]
    check_keys(data_files, required_keys)

    x_test = pd.read_csv(data_files['transformed_x_test_file'])
    y_test = pd.read_csv(data_files['transformed_y_test_file'])

    if model_uri_ is None:
        # task_instance = kwargs.get('task_instance')

        if task_instance is None:
            ValueError("task_instance is required, ensure you are calling this function "
                       "from an airflow task and after a training run.")

        # Get the returns of the 'train_model' function, the task_id of which was named 'train_model' in airflow
        _, model_uri = task_instance.xcom_pull(task_ids='train_model')
    else:
        model_uri = model_uri_

    model = get_model_from_uri(model_uri)
    y_pred = get_predictions(x_test, model)

    val_accuracy, val_f1 = get_val_performance(y_test, y_pred)

    metrics = {
        'val_accuracy': val_accuracy, 'val_f1': val_f1,
    }

    return metrics, y_pred
