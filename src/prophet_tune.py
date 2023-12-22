import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# TS
from prophet import Prophet

# User Imports
import util

# modeling
import optuna
from sklearn.metrics import mean_squared_error as mse
import joblib

load = util.read_load("./data/load_hist_data.csv")
weather = util.read_weather("./data/weather_data.csv")
weather_features = util.featurize_weather(
    weather, lags=[24]
)  # 24 hours = 1 day lagged weather
mod_data = util.create_mod_data(load, weather_features)


from sktime.forecasting.model_selection import temporal_train_test_split

inference_data = mod_data[mod_data.ds >= "2008-01-01"]
mod_data = mod_data[mod_data.ds < "2008-01-01"]

train_data, test_data = temporal_train_test_split(mod_data, test_size=1 / 3)
tune_data, test_data = temporal_train_test_split(test_data, test_size=1 / 3)


def objective(trial):
    params = {
        "changepoint_prior_scale": trial.suggest_float(
            "changepoint_prior_scale", 0.01, 10
        ),
        "seasonality_prior_scale": trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10
        ),
        "seasonality_mode": trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        ),
        "dow_0_prior_scale": trial.suggest_float("dow_0_prior_scale", 0.01, 10),
        "dow_1_prior_scale": trial.suggest_float("dow_1_prior_scale", 0.01, 10),
        "dow_2_prior_scale": trial.suggest_float("dow_2_prior_scale", 0.01, 10),
        "dow_3_prior_scale": trial.suggest_float("dow_3_prior_scale", 0.01, 10),
        "dow_4_prior_scale": trial.suggest_float("dow_4_prior_scale", 0.01, 10),
        "dow_5_prior_scale": trial.suggest_float("dow_5_prior_scale", 0.01, 10),
        "dow_6_prior_scale": trial.suggest_float("dow_6_prior_scale", 0.01, 10),
    }
    print(params)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # added per dow
        # mcmc_samples=300,
        # seasonality_mode="multiplicative",  # "additive",  # "multiplicative",
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
    )

    for i in range(7):
        dow_prior = params[f"dow_{i}_prior_scale"]
        m.add_seasonality(
            name=f"daily_dow{i}",
            period=1,
            fourier_order=4,
            condition_name=f"dow_{i}",
            prior_scale=dow_prior,
        )

    m.add_country_holidays(country_name="US")
    m.add_regressor("max_station_temp")
    m.add_regressor("min_station_temp")
    m.add_regressor("mean_station_temp")
    m.add_regressor("lag_24__min_station_temp")
    m.add_regressor("lag_24__max_station_temp")
    m.add_regressor("lag_24__mean_station_temp")

    m.fit(train_data)
    preds = m.predict(tune_data)
    rmse = mse(tune_data["y"], preds["yhat"], squared=False)
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)

study.best_params
joblib.dump(study, "./models/prophet_study.pkl")
