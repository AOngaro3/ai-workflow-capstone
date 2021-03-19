
import os
import sys
import joblib
import json
import re
import shutil
import time
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from pandas_profiling import ProfileReport


from scripts.utils_data import *
from scripts.logger import *


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.decomposition import PCA



register_matplotlib_converters()
from pathlib import Path
#from scripts import ROOT_DIR

ROOT_DIR = Path(__file__).parent.parent

sys.path.append(os.path.abspath(ROOT_DIR))

RAW_DATA_URL = "https://raw.githubusercontent.com/aavail/ai-workflow-capstone/master/"

DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")

MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

def _read_and_clean(t = "training"):
    
    assert t in ("training","production"), "t must be training or production"
    
    if t == "training":
        train_data = pd.read_csv(DATA_DIR+"/df_training.csv")

        train_data = train_data[train_data["price"] >= 0]
        
        df, country_ts, ts_tot = extract_all_datasets(train_data)
        
       
        
    else: 
        
        prod_data = pd.read_csv(DATA_DIR+"/df_production.csv")

        prod_data = prod_data[prod_data["price"] >= 0]
        
        df, country_ts, ts_tot = extract_all_datasets(prod_data)
    
    return df, country_ts, ts_tot
        
    

def add_supervised_target(df, hm_days):
    result_df = df.copy()

    target = []

    for day in result_df.index:
        start = day + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=hm_days)

        rev_next_days = df["revenue"].loc[start:end].sum()

        target.append(rev_next_days)

    result_df["target"] = target

    return result_df

def rolling_fun(df, window, nome_campo, parametro, min_periods=1):

    
    
    if parametro == 1:
        v = df[nome_campo].rolling(window=window, min_periods=min_periods).mean().values
    elif parametro == 2:
        v = df[nome_campo].rolling(window=window, min_periods=min_periods).std().values
    elif parametro == 3:
        v = df[nome_campo].rolling(window=window, min_periods=min_periods).max().values
    elif parametro == 4:
        v = df[nome_campo].rolling(window=window, min_periods=min_periods).min().values
    elif parametro == 5:
        v = df[nome_campo].rolling(window=window, min_periods=min_periods).sum().values
    else:
        v = df[nome_campo].rolling(window=window, min_periods=min_periods).mean().values
    return v



def lag_fun_tot(df, lista_nome_campo, lista_lag):
  
    i = 0
    for nome_campo in lista_nome_campo:
        for lag in lista_lag:
            if i == 0:
                dg = df.copy()
                v = df[nome_campo].shift(lag)
                #v = lag_fun(dg, key, nome_campo, lag)
                dg[nome_campo + '_' + str(lag)] = v
                i = i + 1
            else:
                v = df[nome_campo].shift(lag)
                dg[nome_campo + '_' + str(lag)] = v
                i = i + 1
    return dg


def rolling_fun_tot(df, lista_window, lista_nome_campo, lista_parametro, min_periods=1):
   
    dg = df.copy()
    l = []
    for nome_campo in lista_nome_campo:
        for parametro in lista_parametro:
            for window in lista_window:
                if parametro == 1:
                    v = rolling_fun(dg, window, nome_campo, 1)
                    dg[nome_campo + '_' + 'mean' + '_' + str(window)] = v
                    l.append(nome_campo + '_' + 'mean' + '_' + str(window))

                elif parametro == 2:
                    v = rolling_fun(dg, window, nome_campo, 2)
                    dg[nome_campo + '_' + 'std' + '_' + str(window)] = v
                    l.append(nome_campo + '_' + 'std' + '_' + str(window))

                elif parametro == 3:
                    v = rolling_fun(dg, window, nome_campo, 3)
                    dg[nome_campo + '_' + 'max' + '_' + str(window)] = v
                    l.append(nome_campo + '_' + 'max' + '_' + str(window))

                elif parametro == 4:
                    v = rolling_fun(dg, window, nome_campo, 4)
                    dg[nome_campo + '_' + 'min' + '_' + str(window)] = v
                    l.append(nome_campo + '_' + 'min' + '_' + str(window))
                
                if (nome_campo + '_' + 'max' + '_' + str(window) in dg.columns) & (nome_campo + '_' + 'min' + '_' + str(window) in dg.columns):
                    dg['range' + '_' + nome_campo + '_' + str(window)] = dg[nome_campo + '_' + 'max' + '_' + str(window)] 
                    - dg[nome_campo + '_' + 'min' + '_' + str(window)]
                    l.append('range' + '_' + nome_campo + '_' + str(window))
    return l,dg
    

def create_supervised_features(df,campo=["revenue"],lag=[1,2,3],window_size=[3],functions=[1,2,3,4]):
    
    result_df = df.copy()
    
    result_df=lag_fun_tot(df,campo,lag)


# Costruzione variabili rolling [mean, max, min, range] a 3,7,15,30 gg

    lista_rolling,result_df=rolling_fun_tot(result_df,window_size,
                                                          campo,
                                                          functions,
                                                          min_periods=1)
    
    return result_df


def supervised_features_and_target(df,hw_days=30):
    
    result_df = df.copy()
    
    result_df = add_supervised_target(result_df.set_index("date"),hw_days)
    
    result_df = create_supervised_features(result_df)
    
    return result_df
    
    
def df_to_model(mode="training",country=None):
    df, country_ts, ts_tot = _read_and_clean(t = mode)
    
    if country == None:
        ts_tot = ts_tot.drop("year_month",axis=1)
    
        result_df = supervised_features_and_target(ts_tot)
    else: 
        assert country in ('United Kingdom','EIRE','Germany','France','Norway','Spain','Hong Kong','Portugal','Singapore','Netherlands'), "country must be one of the top 10"
        country_ts = country_ts.drop(["year_month","country"],axis=1)
        result_df = supervised_features_and_target(ts_tot)
    
    
    return result_df.dropna()


def split_train_test(df, training_perc=0.8, hm_days=30, verbose=0):
    first_date_training = df.index.min()
    last_date_training = first_date_training + pd.Timedelta(days=int(len(df) * training_perc))
    first_date_testing = last_date_training + pd.Timedelta(days=1)
    last_date_testing = df.index.max()

    training_data = df.loc[first_date_training:last_date_training, :]
    testing_data = df.loc[first_date_testing:last_date_testing, :]

    if verbose > 0:
        print(first_date_training, last_date_training, first_date_testing, last_date_testing)

    # correction for the supervised problem: shrink dataset by removing the last hm_days rows,
    # because future revenue data is insufficient to compute the next hm_days days of revenue
    training_data = training_data.iloc[:-hm_days].copy()
    testing_data = testing_data.iloc[:-hm_days].copy()

    if verbose > 0:
        print(training_data.shape, testing_data.shape)

    target_column = "target"
    X_train, y_train = training_data.drop(target_column, 1).values, training_data[target_column].values
    X_test, y_test = testing_data.drop(target_column, 1).values, testing_data[target_column].values

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    return X, y, X_train, y_train, X_test, y_test


def create_train_test(country=None):
    
    sup_df = df_to_model("training",country=country)

    X, y, X_train, y_train, X_test, y_test = split_train_test(sup_df)

    return X, y, X_train, y_train, X_test, y_test

def create_pipe(X_train, y_train,model,scaler,pca,cv=5,mode="test"):
    
    if scaler == "Standard":
        scaler = StandardScaler()
    
    if model == "XGBoost":
        model = XGBRegressor()
        param_grid = {
            'model__n_estimators': [50, 100, 200, 500],
            'model__learning_rate': [0.01, 0.05,0.1],
        }
    elif model == "RandomForest":
        model = RandomForestRegressor()
        param_grid = {
            'model__n_estimators': [50, 100, 200, 500],
            'model__criterion': ["mse", "mae"],
        }
    
    if pca:
        pipe = Pipeline([('scaler', scaler),("PCA",PCA(n_components=4)), ('model', model)])
    else:
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv)
    
    search.fit(X_train, y_train)
    
    pipe = search.best_estimator_
    
    #print("Best parameter (CV score = %0.3f):" % search.best_score_)
    #print(search.best_params_)
    
    pipe.fit(X_train, y_train)
    
    return pipe
    

    
def evaluate_pipe(pipe, X_test, y_test, plot_eval, plot_avg_threshold=np.inf):
    test_predictions = pipe.predict(X_test)

    test_mae = round(mean_absolute_error(test_predictions, y_test), 2)
    test_rmse = round(mean_squared_error(test_predictions, y_test, squared=False), 2)
    test_avg = round(np.mean([test_mae, test_rmse]), 2)

    if (plot_eval is True) & (test_avg < plot_avg_threshold):
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        title = f"{pipe.steps} \n test_mae: {test_mae}, test_rmse:{test_rmse}, test_avg:{test_avg}"
        ax.set_title(title)
        ax.plot(test_predictions, label="pred")
        ax.plot(y_test, label="true")
        ax.legend()
        plt.show(block=False)

    return test_mae, test_rmse, test_avg

def find_best_model(plot=False,printt=False,ret=False,country=None):
    
    X, y, X_train, y_train, X_test, y_test = create_train_test(country=country)
    mae_rif = 1000000000000
    diz_pipe = {"pca":True,"model":"","mae":mae_rif}
    for j in [True,False]:
        for i in ["XGBoost","RandomForest"]:
            pipe = create_pipe(X_train,y_train,i,"Standard",j)
            test_mae, test_rmse, test_avg= evaluate_pipe(pipe,X_test, y_test, plot_eval=plot)
            if printt:
                print(f"Model:{i},Scaler:{j}: mae:{test_mae},rmse:{test_rmse}")
            if test_mae<mae_rif:
                mae_rif = test_mae
                diz_pipe["pca"] = j
                diz_pipe["model"] = i
                diz_pipe["mae"] = test_mae
    if ret == True:
        return X, y, diz_pipe, pipe
                

def train_model(country, test=False):

    start_time = time.time()

    if country == "null":
        country = None

    X,y,diz_pipe,pipe = find_best_model(plot=False,printt=False,ret=True)
    
    model = create_pipe(X,y,diz_pipe["model"],"Standard",diz_pipe["pca"])
    
    best_mae = diz_pipe["mae"]
    print(f"Best score on test set: {best_mae}")
    best_params = model.get_params
    model_type = diz_pipe["model"]

    model_name = f"supervised_model_{model_type}_{country}"
    #params = model.params
    
    if test is False:
        model_name,version = save_model(model,best_params, model_name,test)
    else:
        model_name,version = save_model(model,best_params, model_name,test)

    end_time = time.time()
    runtime = end_time-start_time

    update_train_log(diz_pipe["mae"], runtime, version, test)

    return model_name

def model_predict(starting_dates,model_version, model, test=False, mode="test",modality = "training"):

    
    start_time = time.time()

    df = df_to_model(mode=modality,country=None) 
    
    starting_dates = [pd.Timestamp(sd) for sd in starting_dates]

    if any(sd not in df.index for sd in starting_dates):
        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")
        raise KeyError(f"Acceptables dates range from {start} to {end}.")

    predictions = []
    for sd in starting_dates:
        x = df.drop("target",axis=1).loc[sd].values
        x = x.reshape(1, -1)
        prediction = model.predict(x)
        predictions.append(prediction)

    end_time = time.time()
    runtime = end_time - start_time

    update_predict_log(predictions, starting_dates, runtime, model_version, test)

    return predictions

    
    
def save_model(model,model_params,model_name,test):
      
    # find the model version name
    all_files_in_models = os.listdir(MODEL_DIR)
    all_model_names = [file.replace(".joblib", "") for file in all_files_in_models if file.endswith(".joblib")]
    version_numbers = [int(_model_name.split("_")[-1]) for _model_name in all_model_names]
    if len(version_numbers) == 0:
        new_version_number = "0"
    else:
        new_version_number = str(max(version_numbers) + 1)
    
    if test:
        model_name = "TEST_"+model_name + "_" + new_version_number
    else:
        model_name = model_name + "_" + new_version_number

    model_saving_path = os.path.join(MODEL_DIR, model_name + ".joblib")
    params_saving_path = os.path.join(MODEL_DIR, model_name + ".json")
    
    joblib.dump(model, model_saving_path)

    # save model params
    #with open(params_saving_path, 'w') as f:
    #    json.dump(model_params, f)

    print(f"Model saved in {MODEL_DIR}.")
    
    return model_name, new_version_number

def load_model(model_name=None, country_name=None):
    if model_name == None:
        model_name = find_last_model(country_name)

    model_name = model_name.replace(".joblib", "")
    model_loading_path = os.path.join(MODEL_DIR, model_name + ".joblib")

    model_version = model_name.replace(".joblib", "").split("_")[-1]
    # load model
    loaded_model = joblib.load(model_loading_path)


    return loaded_model, model_name, model_version

def find_last_model(country_name):

    if country_name is None:
        country_name = "None"

    all_files_in_models = os.listdir(MODEL_DIR)
    all_models = [file for file in all_files_in_models if (file.endswith(".joblib") and country_name in file)]
    all_models = sorted(all_models, key=lambda x: int(x.replace(".joblib", "").split("_")[-1]))

    if len(all_models) == 0:
        raise FileNotFoundError

    best_model_name = all_models[-1]
    return best_model_name


if __name__ == "__main__":
    """
    basic test procedure for model.py
    """
    ## train the model
    print("TRAINING MODELS")
    DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")
    MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

    model_name = train_model(country=None,test=True)

    ## load the model
    print("LOADING MODELS")
    loaded_model, model_name, model_version = load_model(model_name=model_name)
    print("... models loaded:")

    ## test predict    
    country = None
    param_dim = "small"
    testing_dates = ["2018-01-25"]
    
    print(f"Testing the model on {testing_dates}")
    prediction = model_predict(testing_dates,model_version=model_version, model=loaded_model,test=True)
    print(prediction)