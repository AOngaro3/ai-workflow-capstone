import pandas as pd
import numpy as np

from scripts.utils_models import *

n = 20
start = "2019-01"
end = "2019-03"
df = pd.DataFrame(data=np.random.random((n, 3)),
                  index=pd.date_range(start, end, periods=n))
df.columns = ["streams", "views", "revenue"]

def test_create_supervised_features(): 
    res_df = create_supervised_features(df)
    cond = res_df.shape[0] != 0
    assert cond


def test_test_split_train_test():
    sup_df = df_to_model("training",country=None)

    X, y, X_train, y_train, X_test, y_test = split_train_test(sup_df)

    cond_train = X_train.shape[0] + X_test.shape[0] == X.shape[0]
    cond_test = y_train.shape[0] + y_test.shape[0] == y.shape[0]

    assert cond_train and cond_test