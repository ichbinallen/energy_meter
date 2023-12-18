# Basics
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Time Series



def read_load(fn):
    df = pd.read_csv(fn)
    df['Date'] = pd.to_datetime(df['Date'])
    df['ds'] = pd.to_datetime(df.apply(
        lambda row: f"{row['Date'].strftime('%Y-%m-%d')}T{str(row['Hour']-1)}:00:00",
        axis=1
    ))
    df = df[['ds', 'Load']]
    
    # Make sure there are no skipped timestamps
    df.set_index('ds', inplace=True)
    
    # remove duplicated ds from time daylight savings
    df = df[~df.index.duplicated()]
    df.asfreq('H')
    # df.reset_index(inplace=True)
    
    return df


def read_weather(fn):
    df = pd.read_csv(fn)
    df['Date'] = pd.to_datetime(df['Date'])
    df['ds'] = pd.to_datetime(df.apply(
        lambda row: f"{row['Date'].strftime('%Y-%m-%d')}T{str(row['Hour']-1)}:00:00",
        axis=1
    ))
    
    df = df[['ds', 'Station ID', 'Temperature']]
    df.rename(columns={'Station ID': 'station_id', 'Temperature': 'temperature'}, inplace=True)
    
    df = df[~df.duplicated(subset=['ds', 'station_id'])]
    
    return df



def featurize_weather(weather):
    weather_features = weather.groupby(['ds']).agg(
        min_station_temp = ('temperature', np.min),
        max_station_temp = ('temperature', np.max),
        mean_station_temp = ('temperature', np.mean)
    ).reset_index()
    
    return weather_features




def slider_plot(data, x, y):
    ''' Create a plotly lineplot with range slider '''
    # Create figure
    fig = go.Figure()
    # Show line
    fig.add_trace(go.Scatter(x=list(data[x]), y=list(data[y])))
    # Add range slider
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    
    fig.show()
