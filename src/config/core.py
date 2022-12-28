from pathlib import Path
import os
import sys
sys.path.append(os.path.relpath('..'))

import src

SRC_ROOT = Path(src.__file__).resolve().parent
ROOT = SRC_ROOT.parent
DATASET_DIR = ROOT / "data"

DATA1 = DATASET_DIR / "input_data/telco_customer_churn_1.csv"
DATA2 = DATASET_DIR / "input_data/telco_customer_churn_2.csv"

RANDOM_STATE = 0
TEST_SIZE = 0.2

VARS_TO_DROP = ['customerID']
CAT_VARS_ONEHOT = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
CAT_VARS_ORDINAL_ARBITARY = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                             'Contract', 'PaymentMethod']
NUM_VARS_YEO_YOHNSON = ['TotalCharges']
TARGET = 'Churn'
VAR_REPLACE_EMPTY_DATA = ['TotalCharges']

