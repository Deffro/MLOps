import mlflow
from mlflow.tracking.client import MlflowClient

from src.modeling.model_validation import evaluate_model
from mlflow.exceptions import MlflowException
from loguru import logger


def check_if_registered_model_exists(reg_model):
    """
    For airflow: Check if the requested registered model exists in order to proceed to the next task_id
    reg_model (str): The requested registered model. Example: "LR/2"
    """
    if len(reg_model.split('/')) != 2:
        raise ValueError("reg_model should be in the format: 'LR/2'.")
    model_name = reg_model.split('/')[0]

    reg_model_v1 = model_name + '/1'
    registered_model_v1_uri = f"models:/{reg_model_v1}"
    registered_model_uri = f"models:/{reg_model}"

    v1_model_exists = False
    try:
        # First check if the version 1 of the requested model exists
        mlflow.pyfunc.load_model(model_uri=registered_model_v1_uri)
        logger.info(f"The Version 1 of the requested model {reg_model} exists!")
        v1_model_exists = True
    except MlflowException:
        logger.info(f"There is no Version 1 of the requested model {reg_model}.")
        return 'register_model'

    if v1_model_exists is True:
        try:
            # First check if the requested model and version exist
            reg_model = mlflow.pyfunc.load_model(model_uri=registered_model_uri)
            logger.info(f"Registered model {reg_model} exists!")
            return 'register_model_by_comparison'
        except MlflowException:
            logger.info(f"There is no registered model yet with model uri {reg_model}")
            return 'stop'


def get_register_model(model_name, model_uri=None):

    mv = mlflow.register_model(model_uri, model_name)
    logger.success(f"Model {model_name} registered.")

    return mv


def register_model(model_name, model_uri_=None, **kwargs):
    """
    Register a model to the MLflow Model Registry
    model_name (str): The name of the model to be registered in mlflow
    model_uri_: Pass a mlflow uri if running without using airflow
    When using airflow, this uri is passed through task_instance.xcom_pull(task_ids='train_model')
    """

    if model_uri_ is None:
        task_instance = kwargs.get('task_instance')

        if task_instance is None:
            ValueError("task_instance is required, ensure you are calling this function "
                       "from an airflow task and after a training run.")

        # Get the returns of the 'train_model' function, the task_id of which was named 'train_model' in airflow
        _, model_uri = task_instance.xcom_pull(task_ids='train_model')
    else:
        model_uri = model_uri_

    get_register_model(model_name, model_uri)


def register_model_by_comparison(data_files, registered_model_uri,  model_name,
                                 push_to_production=False, **kwargs):
    """
    Compare a newly created model to an already registered model. If the new model has
    greater f1 score, register it.
    param data_files (dict): A dictionary of data file paths.
    registered_model_uri: The already registered model. Example: "LR/2"
    model_name: The name of the model to register. Example: "LR"
    push_to_production: True or False
    registered_model_uri: Pass a mlflow uri if running without using airflow
    When using airflow, this uri is passed through task_instance.xcom_pull(task_ids='train_model')
    """

    task_instance = kwargs.get('task_instance')

    if task_instance is None:
        ValueError("task_instance is required, ensure you are calling this function "
                   "from an airflow task and after a training run.")

    # Get the returns of the 'train_model' function, the task_id of which was named 'train_model' in airflow
    _, new_model_to_compare_uri = task_instance.xcom_pull(task_ids='train_model')

    registered_model_uri = f"models:/{registered_model_uri}"
    metrics_registered_model, _ = evaluate_model(data_files, model_uri_=registered_model_uri)
    metrics_model_to_compare_uri, _ = evaluate_model(data_files, model_uri_=new_model_to_compare_uri, task_instance=task_instance)

    if metrics_model_to_compare_uri['val_f1'] >= metrics_registered_model['val_f1']:
        logger.info(f"Model {new_model_to_compare_uri} superior to {registered_model_uri}. "
                    f"({metrics_model_to_compare_uri['val_f1']} vs {metrics_registered_model['val_f1']})")
        logger.info("Registering new model.")
        mv = get_register_model(model_name, registered_model_uri)
        if push_to_production is True:
            promote_model_to_production(f"{mv.name}/{mv.version}")
    else:
        logger.info(f"Model {new_model_to_compare_uri} inferior to {registered_model_uri}. "
                    f"({metrics_model_to_compare_uri['val_f1']} vs {metrics_registered_model['val_f1']})")


def promote_model_to_production(reg_model):
    """
    Promote current model to production after archiving the current production
    reg_model: Example "LR/2"
    """
    client = MlflowClient()

    if len(reg_model.split('/')) != 2:
        raise ValueError("reg_model should be in the format: 'LR/2'.")
    model_name = reg_model.split('/')[0]
    model_version = reg_model.split('/')[1]

    # Get all registered models of given model_name tagged with "Production"
    try:
        prod_model_versions = client.get_latest_versions(
            model_name, stages=["Production"])
    except:
        prod_model_versions = []

    # Change their tags to "Archived"
    for prod_model_version in prod_model_versions:
        print(f"Archiving model: {prod_model_version}")
        client.transition_model_version_stage(
            name=prod_model_version.name,
            version=prod_model_version.version,
            stage="Archived"
        )

    print(f"Promoting model: {model_name} as version: {model_version}")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )
