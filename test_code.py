import os
from src.processing.data_transformation import read_data, split_train_test, preprocess_data
from src.modeling.model_train import train_model
from src.modeling.model_push import register_model, register_model_by_comparison, promote_model_to_production
from src.modeling.model_validation import evaluate_model
from src.config.core import DATA1

model_name = 'XGB'
experiment_name = "churn_prediction"

_data_dir = "/data"
# Every data file that is created through code
_output_data_dir = _data_dir + '/output_data'
if not os.path.exists(_output_data_dir):
    os.makedirs(_output_data_dir)
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

read_data(_data_files)
split_train_test(_data_files)

preprocess_data(_data_files)

# Train the model. Also track validation performance with the optional parameters x_test and y_test
run_id, model_uri = train_model(_data_files, experiment_name, model_name,
                                track_cv_performance=True)

# Load the trained model and evaluate
metrics, y_pred = evaluate_model(_data_files, model_uri_offline=model_uri)
print(metrics)

# Register the model
# register_model(model_uri, model_name)

# Promote a registered model to production
reg_model = "LR/2"
# promote_model_to_production(reg_model)

# Load a registered model and evaluate
# metrics_prod, y_pred_prod = evaluate_model(_data_files)
# print(metrics_prod)

# Compare current model with an already registered one and register current if superior
# Push it to production if desired
register_model_by_comparison(_data_files, reg_model,  model_name, push_to_production=True)


