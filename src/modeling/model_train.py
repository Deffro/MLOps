# Use all training data and train a model on them

import mlflow
from sklearn.linear_model import LogisticRegression


def get_model(model_name='LR'):
    accepted_models = ['LR', 'XGB']
    if model_name not in accepted_models:
        raise ValueError(f"'model name' should be in: {accepted_models}.")

    model = None
    if model_name == 'LR':
        C = 0.8
        iterations = 200
        model = LogisticRegression(C=C, max_iter=iterations)

    return model


def train_model(x_train, y_train):

    clf = get_model()

    # MLflow: tell MLflow where the model tracking server is
    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    # MLflow: experiment name
    _experiment_name = "churn-prediction"
    mlflow.set_experiment(_experiment_name)

    with mlflow.start_run() as run:
        # MLflow: print run specific info
        run_id = run.info.run_id
        print(f"\nActive run_id: {run_id}")

        # Train
        clf.fit(x_train, y_train)

        # MLflow log the model
        mlflow.sklearn.log_model(clf, "logistic_regression_model")
        model_uri = mlflow.get_artifact_uri("logistic_regression_model")

    return run_id, model_uri




