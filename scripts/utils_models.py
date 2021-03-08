
import os
import sys
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

from .utils_data import *


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV


register_matplotlib_converters()
from pathlib import Path
#from scripts import ROOT_DIR

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_URL = "https://raw.githubusercontent.com/aavail/ai-workflow-capstone/master/"
DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")

def _read_and_clean(t = "training"):
    
    assert t in ("training","production"), "t must be training or production"
    
    if t == "training":
        train_data = pd.read_csv("./data/datasets/df_training.csv")

        train_data = train_data[train_data["price"] >= 0]
        
        df, country_ts, ts_tot = extract_all_datasets(train_data)
        
       
        
    else: 
        
        prod_data = pd.read_csv("./data/datasets/df_production.csv")

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
    
    
def df_to_model(mode="training"):
    df, country_ts, ts_tot = _read_and_clean(t = mode)
    
    ts_tot = ts_tot.drop("year_month",axis=1)
    
    result_df = supervised_features_and_target(ts_tot)
    
    return result_df


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


def create_train_test():
    
    sup_df = df_to_model("training")

    X, y, X_train, y_train, X_test, y_test = split_train_test(sup_df)

    return X, y, X_train, y_train, X_test, y_test





if __name__ == "__main__":

    run_start = time.time() 
    
    # = os.path.join("..","data","cs-train")
    print("...fetching data")

    df_train = fetch_data(RAW_DATA_URL,train=True)
    prod_df = fetch_data(RAW_DATA_URL,train=False)

    df_train.to_csv(os.path.join(DATA_DIR, "df_training.csv"), index=False)
    prod_df.to_csv(os.path.join(DATA_DIR, "df_production.csv"), index=False)
    
    #ts_all = fetch_ts(DATA_DIR,clean=False)
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))

    #for key,item in ts_all.items():
    #    print(key,item.shape)