# Basics
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Time Series
from sktime.transformations.series.lag import Lag


def read_load(fn):
    """
    read energy load data from csv
    """

    df = pd.read_csv(fn)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ds"] = pd.to_datetime(
        df.apply(
            lambda row: f"{row['Date'].strftime('%Y-%m-%d')}T{str(row['Hour']-1)}:00:00",
            axis=1,
        )
    )
    df = df[["ds", "Load"]]
    df.rename(columns={"Load": "y"}, inplace=True)

    # Make sure there are no skipped timestamps
    df.set_index("ds", inplace=True)

    # remove duplicated ds from time daylight savings
    df = df[~df.index.duplicated()]
    df = df.asfreq("H")
    df.interpolate(method="linear", inplace=True)
    df.reset_index(inplace=True)

    return df


def read_weather(fn):
    """
    read weather data from csv
    """
    df = pd.read_csv(fn)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ds"] = pd.to_datetime(
        df.apply(
            lambda row: f"{row['Date'].strftime('%Y-%m-%d')}T{str(row['Hour']-1)}:00:00",
            axis=1,
        )
    )

    df = df[["ds", "Station ID", "Temperature"]]
    df.rename(
        columns={"Station ID": "station_id", "Temperature": "temperature"}, inplace=True
    )

    df = df[~df.duplicated(subset=["ds", "station_id"])]

    return df


def featurize_weather(weather, lags=[24]):
    """
    Extract Useful Weather Features for a global Load forecast

    Note: I am using future weather features to improve forecast accuracy,
          but these would not be available in real applications.  Future
          weather features would need to be foreacasted or seasonal averages

    Take the most extreme weather station, mean of all weather stations, and min weather station for all time points

    If lags is specified, also include lag weather features (e.g. use lags=24 to include temps from yesterday)

    returns: weather features df
    """

    # Aggregate accross station ids
    weather_features = (
        weather.groupby(["ds"])
        .agg(
            min_station_temp=("temperature", "min"),
            max_station_temp=("temperature", "max"),
            mean_station_temp=("temperature", "mean"),
        )
        .reset_index()
    )

    # Make sure data is at Hourly Time Grain
    weather_features.set_index("ds", inplace=True)
    weather_features = weather_features.asfreq("H")
    weather_features.interpolate(inplace=True)

    # Lag Transform Weather featuers
    lag_transformer = Lag(lags)
    lag_weather_features = lag_transformer.fit_transform(weather_features)
    lag_feature_names = list(lag_weather_features.columns)

    weather_features.reset_index(inplace=True)
    lag_weather_features.reset_index(inplace=True)

    # join back all features
    weather_features = weather_features.merge(lag_weather_features, on="ds", how="left")
    for lag_feat in lag_feature_names:
        orig_feat = lag_feat.split("__")[1]
        weather_features[lag_feat] = np.where(
            weather_features[lag_feat].isna(),
            weather_features[orig_feat],
            weather_features[lag_feat],
        )
    return weather_features


def create_mod_data(load, weather):
    """
    Combines load and weather to create modeling data

    params:
    load: pd.DataFrame - df containing ds, y
    weather: pd.DataFrame - df containing ds, weather features, lag weather features, etc

    returns: pd.DataFrame mod_data
    """
    mod_data = load.copy()
    mod_data = mod_data.merge(weather, on="ds", how="outer")

    mod_data["dow"] = mod_data.ds.dt.dayofweek
    mod_data["school_break"] = (
        (mod_data.ds.dt.month.isin([6, 5, 8]))  # Summer Time
        | (
            (mod_data.ds.dt.month == 12) & (mod_data.ds.dt.day >= 18)
        )  # Winter Holiday Dec
        | (
            (mod_data.ds.dt.month == 1) & (mod_data.ds.dt.day <= 7)
        )  # Winter Holiday Jan
    )
    mod_data = pd.get_dummies(mod_data, prefix="dow", prefix_sep="_", columns=["dow"])

    return mod_data


def slider_plot(data, x, y):
    """Create a plotly lineplot with range slider"""
    # Create figure
    fig = go.Figure()
    # Show line
    fig.add_trace(go.Scatter(x=list(data[x]), y=list(data[y])))
    # Add range slider
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

    fig.show()
