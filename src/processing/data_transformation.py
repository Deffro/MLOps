import pandas as pd
from feature_engine.encoding import (
    OrdinalEncoder,
    OneHotEncoder,
)
from feature_engine.transformation import (
    YeoJohnsonTransformer,
)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.config.core import DATA1, RANDOM_STATE, TEST_SIZE, VARS_TO_DROP, TARGET, CAT_VARS_ONEHOT, CAT_VARS_ORDINAL_ARBITARY, NUM_VARS_YEO_YOHNSON, VAR_REPLACE_EMPTY_DATA


def read_data(path=DATA1) -> pd.DataFrame:
    """
    Read a csv from the provided path
    """
    data = pd.read_csv(path)
    return data


def split_train_test(data: pd.DataFrame):
    """
    Split data into train and test set. Test size is declared
    as a constant in the config
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(VARS_TO_DROP+[TARGET], axis=1),
        data[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return x_train, x_test, y_train, y_test


def replace_empty_in_col(data: pd.DataFrame) -> pd.DataFrame:
    """
    In order to convert a string variable that is numeric to float,
    replace empty space value with -1
    """
    for feature in VAR_REPLACE_EMPTY_DATA:
        data[feature] = data[feature].str.replace(' ', '-1').astype(float)

    return data


def fit_categorical_encoders(x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit categorical encoders on train data
    """
    ordinal_encoder_arbitrary = OrdinalEncoder(encoding_method='arbitrary', variables=CAT_VARS_ORDINAL_ARBITARY)
    ordinal_encoder_arbitrary.fit(x_train, y_train)

    onehot_encoder = OneHotEncoder(variables=CAT_VARS_ONEHOT)
    onehot_encoder.fit(x_train)

    cat_encoders = {'ordinal_encoder': ordinal_encoder_arbitrary,
                    'onehot_encoder': onehot_encoder}

    return cat_encoders


def transform_categorical_encoders(x_to_encode: pd.DataFrame, cat_encoders: dict) -> pd.DataFrame:
    """
    Use pre-fitted categorical encoders to transform data
    """
    for encoder in cat_encoders.values():
        x_to_encode = encoder.transform(x_to_encode)

    return x_to_encode


def fit_numerical_transformers(x_train: pd.DataFrame) -> dict:
    """
    Fit numerical transformers on train data
    """
    yeo_transformer = YeoJohnsonTransformer(variables=NUM_VARS_YEO_YOHNSON)
    yeo_transformer.fit(x_train)

    num_transformers = {'yeo_transformer': yeo_transformer}

    return num_transformers


def transform_numerical_transformers(x_to_transform: pd.DataFrame, num_transformers: dict) -> pd.DataFrame:
    """
    Use pre-fitted numerical transformers to transform data
    :return:
    """
    for transformer in num_transformers.values():
        x_to_transform = transformer.transform(x_to_transform)

    return x_to_transform


def fit_target_encoder(y_train: pd.Series):
    """
    Fit an encoder for the target variable
    """
    le = LabelEncoder()
    le.fit(y_train)

    return le


def transform_target_encoder(encoder, y_to_transform: pd.Series) -> pd.Series:
    """
    Use a pre-fitted encoder to transform a Series
    """
    y_to_transform = encoder.transform(y_to_transform)

    return y_to_transform


def fit_data_scaler(x_train: pd.DataFrame):
    """
    Fit a scaler to normalize data
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train)

    return min_max_scaler


def transform_data_scaler(scaler, x_to_transform: pd.DataFrame) -> pd.DataFrame:
    """
    Use a ore-fitted scaler to normalise data
    """
    x_to_transform = pd.DataFrame(scaler.transform(x_to_transform), columns=x_to_transform.columns)

    return x_to_transform


def oversample_data(x_train: pd.DataFrame, y_train: pd.Series):
    """
    Create artificial rows so that both classes have equal observations
    """
    x_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(x_train, y_train)

    return x_train, y_train


def preprocess_data(x_train: pd.DataFrame, y_train: pd.Series,
                    x_test=None, y_test=None):
    """
    Pipeline of all preprocessing functions
    """
    x_train = replace_empty_in_col(x_train)
    if x_test is not None:
        x_test = replace_empty_in_col(x_test)

    cat_encoders = fit_categorical_encoders(x_train, y_train)
    x_train = transform_categorical_encoders(x_train, cat_encoders)
    if x_test is not None:
        x_test = transform_categorical_encoders(x_test, cat_encoders)

    num_transformers = fit_numerical_transformers(x_train)
    x_train = transform_numerical_transformers(x_train, num_transformers)
    if x_test is not None:
        x_test = transform_numerical_transformers(x_test, num_transformers)

    target_encoder = fit_target_encoder(y_train)
    y_train = transform_target_encoder(target_encoder, y_train)
    if y_test is not None:
        y_test = transform_target_encoder(target_encoder, y_test)

    scaler = fit_data_scaler(x_train)
    x_train = transform_data_scaler(scaler, x_train)
    if x_test is not None:
        x_test = transform_data_scaler(scaler, x_test)

    x_train, y_train = oversample_data(x_train, y_train)

    return x_train, y_train, x_test, y_test
