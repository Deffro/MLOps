from src.processing.data_transformation import read_data, split_train_test, preprocess_data
from src.modeling.model_train import train_model
from src.modeling.model_push import register_model, register_model_by_comparison, promote_model_to_production
from src.modeling.model_validation import evaluate_model

data = read_data()
x_train, x_test, y_train, y_test = split_train_test(data)

x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test=x_test, y_test=y_test)

# Train the model. Also track validation performance with the optional parameters x_test and y_test
model_name = 'XGB'
run_id, model_uri = train_model(x_train, y_train, model_name=model_name,
                                x_test=x_test, y_test=y_test)

# Load the trained model and evaluate
metrics, y_pred = evaluate_model(x_test, y_test, model_uri)
print(metrics)

# Register the model
# register_model(model_uri, model_name)

# Promote a registered model to production
reg_model = "LR/2"
# promote_model_to_production(reg_model)

# Load a registered model and evaluate
# metrics_prod, y_pred_prod = evaluate_model(x_test, y_test, reg_model_uri)
# print(metrics_prod)

# Compare current model with an already registered one and register current if superior
# Push it to production if desired
register_model_by_comparison(registered_model_uri=reg_model, new_model_to_compare_uri=model_uri,
                             model_name=model_name, x_test=x_test, y_test=y_test,
                             push_to_production=True)


