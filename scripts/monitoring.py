
import argparse
import sys 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
sys.path.append(os.path.abspath(".."))
from scripts.utils_models import *

def find_production_dates():
    sup_df = df_to_model(mode="production")

    start = sup_df.index.min()
    end = sup_df.index.max()

    return start, end, sup_df


def evaluate_production_performance(df, hm_days):

    df = df.iloc[:-hm_days]

    return df["target"]


def plot_production_results(true, preds):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].set_title("true vs pred, production data")
    ax[0].plot(preds, label="preds")
    ax[0].plot(true, label="true")
    ax[0].set_ylabel("revenue")
    ax[0].legend()

    ax[1].set_title("absolute error, production data")
    ax[1].plot(abs(preds - true))
    ax[1].set_ylabel("error")
    plt.show()


def compute_production_error(true, preds):
    prod_mae = mean_absolute_error(true, preds)
    prod_rmse = mean_squared_error(true, preds, squared=False)

    prod_error = np.mean([prod_mae, prod_rmse])
    prod_error = round(prod_error, 2)

    return prod_error


def monitor(country, hm_days):
    model, model_name,version = load_model(country_name=country)
    #model_params = load_model_params(model_name)

    start, end, sup_df = find_production_dates()

    starting_dates = list(pd.date_range(start, end))

    prod_preds = model_predict(starting_dates,version, model, test=False, mode="test",modality="production")

    prod_true = evaluate_production_performance(sup_df, hm_days)
    cleaned_prod_preds = pd.Series(prod_preds[:-hm_days],
                                   index=prod_true.index).apply(lambda x: x[0])

    prod_error = compute_production_error(prod_true, cleaned_prod_preds)

    print(f"Error on production data: {prod_error}")

    plot_production_results(prod_true, cleaned_prod_preds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='monitor production data')
    parser.add_argument('-c', '--country', required=True, help='name of the country or None')
    parser.add_argument('-d', '--hm_days', default=30, type=int, help='how many days in the future to predict')

    args = parser.parse_args()

    monitor(args.country, args.hm_days)