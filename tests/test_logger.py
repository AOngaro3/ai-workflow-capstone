from scripts.logger import update_train_log, update_predict_log
import os
import pandas as pd
from pathlib import Path
#from scripts import ROOT_DIR
ROOT_DIR = Path(__file__).parent.parent

LOG_DIR = os.path.join(ROOT_DIR, "logs")

def test_update_train_log():

    eval_test = 30.564
    runtime = 743439832
    model_version = 23
    test = True
    
    update_train_log(eval_test, runtime, model_version, test)

    log_path = os.path.join(LOG_DIR, "train-test.log")

    assert os.path.exists(log_path)


def test_update_predict_log():
    y_pred = [35, 76.32, 123.4234]
    starting_dates = ["2017-09-08", "2018-01-01", "2018-02-07"]
    query = [pd.Timestamp(sd) for sd in starting_dates]
    runtime = 743439832
    model_version = 23
    test = True

    update_predict_log(y_pred, query, runtime, model_version, test)

    log_path = os.path.join(LOG_DIR, "predict-test.log")

    assert os.path.exists(log_path)