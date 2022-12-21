from src.processing.data_transformation import read_data, split_train_test, preprocess_data
from src.modeling.model_train import train_model

data = read_data()
x_train, x_test, y_train, y_test = split_train_test(data)

x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test=x_test, y_test=y_test)

run_id, model_uri = train_model(x_train, y_train)





