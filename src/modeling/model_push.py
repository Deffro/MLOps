import mlflow
from mlflow.tracking.client import MlflowClient

from src.modeling.model_validation import evaluate_model
from loguru import logger


def register_model(model_uri, model_name):
    """
    Register a model to the MLflow Model Registry
    """
    mv = mlflow.register_model(model_uri, model_name)
    logger.info(f"Model {model_name} registered.")

    return mv


def register_model_by_comparison(registered_model_uri, new_model_to_compare_uri,  model_name,
                                 x_test, y_test, push_to_production=False):
    """
    Compare a newly created model to an already registered model. If the new model has
    greater f1 score, register it.
    :param registered_model_uri: The already registered model. Example: "models:/LR/2"
    :param new_model_to_compare_uri: The current model_uri as returned from the train_model
    :param model_name: The name of the model to register. Example: "LR"
    :param push_to_production: True or False
    :param x_test: Test data
    :param y_test: Test target
    """
    registered_model_uri = f"models:/{registered_model_uri}"
    metrics_registered_model, _ = evaluate_model(x_test, y_test, registered_model_uri)
    metrics_model_to_compare_uri, _ = evaluate_model(x_test, y_test, new_model_to_compare_uri)

    if metrics_model_to_compare_uri['val_f1'] >= metrics_registered_model['val_f1']:
        logger.info(f"Model {new_model_to_compare_uri} superior to {registered_model_uri}. "
                    f"({metrics_model_to_compare_uri['val_f1']} vs {metrics_registered_model['val_f1']})")
        logger.info("Registering new model.")
        rm = register_model(registered_model_uri, model_name)
        if push_to_production is True:
            promote_model_to_production(f"{rm.name}/{rm.version}")
    else:
        logger.info(f"Model {new_model_to_compare_uri} inferior to {registered_model_uri}. "
                    f"({metrics_model_to_compare_uri['val_f1']} vs {metrics_registered_model['val_f1']})")


def promote_model_to_production(reg_model):
    """
    Promote current model to production after archiving the current production
    :reg_model: Example "LR/2"
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
