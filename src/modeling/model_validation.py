import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score


def get_val_performance(y_true: np.array, y_pred: np.array):
    val_accuracy = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average='macro')

    return val_accuracy, val_f1


def get_cv_performance(x_train: pd.DataFrame, y_train: np.array, model):
    # For CV model should be sklearn and not a loaded trained mlflow model
    if "sklearn" not in str(type(model)):
        raise TypeError("Model should be and sklearn model.")
    cv_accuracy = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_macro').mean()

    return cv_accuracy, cv_f1


def get_model(model_uri):
    model = None
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except:
        print(f"There is no model {model_uri}.")

    return model


def get_predictions(x_test: pd.DataFrame, model):
    y_pred = model.predict(x_test)

    return y_pred


def evaluate_model(x_test: pd.DataFrame, y_test: np.array, model_uri):
    model = get_model(model_uri)
    y_pred = get_predictions(x_test, model)

    val_accuracy, val_f1 = get_val_performance(y_test, y_pred)

    metrics = {
        'val_accuracy': val_accuracy, 'val_f1': val_f1,
    }

    return metrics, y_pred
