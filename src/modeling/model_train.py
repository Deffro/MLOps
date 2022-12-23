# Use all training data and train a model on them

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from loguru import logger

from src.modeling.model_validation import get_cv_performance, get_val_performance, get_predictions, evaluate_model


def get_model(model_name='LR'):
    """
    Define a model and return it
    """
    accepted_models = ['LR', 'XGB']
    if model_name not in accepted_models:
        raise ValueError(f"'model name' should be in: {accepted_models}.")

    model = None
    if model_name == 'LR':
        C = 0.7
        iterations = 200
        model = LogisticRegression(C=C, max_iter=iterations)
    if model_name == 'XGB':
        learning_rate = 0.05
        max_depth = 5
        n_estimators = 200
        model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)

    return model


def train_model(x_train: pd.DataFrame, y_train: np.array, model_name,
                track_cv_performance=True, x_test=None, y_test=None):
    """
    Train a model and track it with MLflow. Return model_uri so
    the model can be available for next steps in the pipeline.
    Enabling 'track_cv_performance' will perform cross validation.
    Passing 'x_test' and 'y_test' will perform evaluation.
    """
    # Get the untrained model
    clf = get_model(model_name)

    # MLflow: tell MLflow where the model tracking server is
    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    # MLflow: experiment name
    _experiment_name = "churn-prediction"
    mlflow.set_experiment(_experiment_name)

    with mlflow.start_run() as run:
        # MLflow: print run specific info
        run_id = run.info.run_id
        logger.info(f"\nActive run_id: {run_id}")

        # Train
        clf.fit(x_train, y_train)

        # MLflow: track model parameters
        mlflow.log_params(clf.get_params())

        # MLflow: track CV performance
        if track_cv_performance is True:
            cv_accuracy, cv_f1 = get_cv_performance(x_train, y_train, clf)
            metrics = {"cv_accuracy": cv_accuracy, "cv_f1": cv_f1}
            mlflow.log_metrics(metrics)

        # MLflow: track performance on validation
        if x_test is not None and y_test is not None:
            y_pred = get_predictions(x_test, clf)
            val_accuracy, val_f1 = get_val_performance(y_test, y_pred)
            metrics = {"val_accuracy": val_accuracy, "val_f1": val_f1}
            mlflow.log_metrics(metrics)

        # MLflow log the model
        mlflow.sklearn.log_model(clf, model_name)
        model_uri = mlflow.get_artifact_uri(model_name)

    return run_id, model_uri




