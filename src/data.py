import pandas as pd
import numpy as np

# TS
from sktime.forecasting.model_selection import temporal_train_test_split


# User Imports
from util import (read_load, read_weather, featurize_weather, create_mod_data)



def main():
    load = read_load("./data/load_hist_data.csv")
    weather = read_weather("./data/weather_data.csv")
    weather_features = featurize_weather(weather, lags=[24]) # 1 day lagged weather
    
    mod_data = create_mod_data(load, weather_features)
    
    train_data, test_data = temporal_train_test_split(mod_data, test_size=1 / 3)
    tune_data, test_data = temporal_train_test_split(test_data, test_size=1 / 3)
    
    train_data.to_csv("./data/modeling/train_data.csv", index=False)
    tune_data.to_csv("./data/modeling/tune_data.csv", index=False)
    test_data.to_csv("./data/modeling/test_data.csv", index=False)
    
if __name__ == '__main__':
    main()