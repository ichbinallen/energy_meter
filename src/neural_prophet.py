import pandas as pd
import numpy as np
import joblib

# TS
from neuralprophet import NeuralProphet, uncertainty_evaluate
from sktime.forecasting.model_selection import temporal_train_test_split

# User Imports
import util

# --------------------------------------------------------------------

# modeling data
load = util.read_load("./data/load_hist_data.csv")
weather = util.read_weather("./data/weather_data.csv")
weather_features = util.featurize_weather(
    weather, lags=[24]
)  # 24 hours = 1 day lagged weather
mod_data = util.create_mod_data(load, weather_features)
mod_data.drop(columns=["school_break"], inplace=True)


inference_data = mod_data[mod_data.ds >= "2008-01-01"]
mod_data = mod_data[mod_data.ds < "2008-01-01"]

train_data, test_data = temporal_train_test_split(mod_data, test_size=1 / 3)
tune_data, test_data = temporal_train_test_split(test_data, test_size=1 / 3)

# --------------------------------------------------------------------

confidence_lvl = 0.90
quantile_list = [
    round(((1 - confidence_lvl) / 2), 2),
    round((confidence_lvl + (1 - confidence_lvl) / 2), 2),
]
method = "naive"
alpha = 1 - confidence_lvl

# Retrain with all data - use last 2 months as calibration data

N_FORECASTS = 14 * 24

deploy_mod = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=1 * 12,
    ar_reg=0.5,
    epochs=20,
    n_forecasts=N_FORECASTS,  # steps ahead to forecast
    quantiles=quantile_list,
)
for i in range(7):
    deploy_mod.add_seasonality(
        name=f"daily_dow{i}",
        period=1,
        fourier_order=4,
        condition_name=f"dow_{i}",
    )
deploy_mod.add_country_holidays(country_name="US")
deploy_mod.add_future_regressor("max_station_temp")
deploy_mod.add_future_regressor("min_station_temp")
deploy_mod.add_future_regressor("mean_station_temp")
deploy_mod.add_future_regressor("lag_24__min_station_temp")
deploy_mod.add_future_regressor("lag_24__max_station_temp")
deploy_mod.add_future_regressor("lag_24__mean_station_temp")

deploy_metrics = deploy_mod.fit(
    df=mod_data,
    freq="H",
)
deploy_metrics.to_csv("./data/deploy_model_metrics.csv", index=False)
print(deploy_metrics)

# joblib.dump(deploy_mod, "./models/deploy_model.joblib")

# --------------------------------------------------------------------
# Conformal Prediction
# --------------------------------------------------------------------



inference_preds = deploy_mod.conformal_predict(
    pd.concat([mod_data, inference_data]),
    calibration_df=test_data,
    alpha=alpha,
    method="naive",
)
inference_preds.to_csv(
    f"./data/inference_preds_nforecasts{N_FORECASTS}.csv", index=False
)
