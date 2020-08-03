from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import round_minutes, holoborodko_diff


def split_sequences(sequences, n_steps):
    x, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def process_data(data, step, ratio):
    values = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    split = int(len(data) * ratio)
    if ratio <= 0.8:
        test_split = int(len(data) * (ratio + 0.1))
    elif ratio == 0.9:
        test_split = int(len(data) * (ratio + 0.05))
    train = values[:split]
    valid = values[split:test_split]
    test = values[test_split:]
    data_train = scaler.fit_transform(train)
    data_valid = scaler.transform(valid)
    data_test = scaler.transform(test)
    x_train, y_train = split_sequences(data_train, step)
    x_valid, y_valid = split_sequences(data_valid, step)
    x_test, y_test = split_sequences(data_test, step)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, scaler


def clean_soil(csv, absolute=False):
    csv = csv[['name', 'time', '69886_rssi', '69886_snr', 'degreesC', 'humidity']]  # without f3c80 gateway
    csv.time = csv.time.astype(str)
    csv['time'] = csv['time'].map(lambda x: x[0:10])
    csv.time = csv.time.astype(int)
    csv['time'] = csv['time'].map(lambda x: datetime.utcfromtimestamp(x))
    csv['degreesC'] = csv['degreesC'].map(lambda x: (x/10)-2)
    csv['humidity'] = csv['humidity'].map(lambda x: (x-220)/11)
    csv.drop(csv[(csv.humidity > 100) | (csv.humidity < 5)].index, inplace=True)
    csv.drop(csv[csv.time.dt.year < 2019].index, inplace=True)
    csv.rename(columns={'degreesC': 'soil_temp', 'humidity': 'soil_humidity'}, inplace=True)
    csv['time'] = csv['time'].dt.floor('Min')
    csv = csv.drop_duplicates('time', keep='last')
    if "Senzor_zemlje_2" in csv.name.values:
        csv.drop(csv.loc['2020-01-07':'2020-01-31'].index, axis=0, inplace=True)
    # elif "Senzor_zemlje" in csv.name.values:
        # csv.drop(csv.loc['2019-12-23':'2020-01-21'].index, axis=0, inplace=True)
    cols = ['69886_rssi', '69886_snr']  # without f3c80 gateway
    if absolute is True:
        for var in cols:
            csv[var] = csv[var].map(lambda x: np.power(10, x/10))
    csv.drop('soil_temp', axis=1, inplace=True)
    csv.time = csv['time'].map(lambda x: round_minutes(x))
    #csv = csv.drop_duplicates('time', keep='last')
    csv.dropna(axis=0, inplace=True)
    csv.reset_index(drop=True, inplace=True)
    csv.set_index('time', drop=True, inplace=True)
    csv.drop(csv.loc['2020-07-01':'2020-07-03'].index, axis=0, inplace=True)
    csv.drop(['name'], axis=1, inplace=True)
    return csv


def clean_air(csv):
    csv.time = csv.time.astype(str)
    csv['time'] = csv['time'].map(lambda x: x[0:10])
    csv.time = csv.time.astype(int)
    csv['time'] = csv['time'].map(lambda x: datetime.utcfromtimestamp(x))
    csv['time'] = csv['time'].dt.floor('Min')
    csv = csv.drop_duplicates('time', keep='last')
    if "Senzor_zraka" in csv.name.values:
        csv.drop(["ae05e_time", "f3c80_time"], axis=1, inplace=True)
        csv.rename(columns={'degreesC': 'air_temp', 'humidity': 'air_humidity'}, inplace=True)
        csv.drop(csv[(csv.air_humidity > 100) | (csv.air_humidity < 5)].index, inplace=True)
        csv.drop(csv[(csv.air_temp > 50) | (csv.air_temp < 0)].index, inplace=True)
        csv.drop(csv.index[0], axis=0, inplace=True)
    elif "DHMZ_new" in csv.name.values:
        csv.rename(columns={'Tlak': 'pressure'}, inplace=True)
    csv.time = csv['time'].map(lambda x: round_minutes(x))
    #csv = csv.drop_duplicates('time', keep='last')
    csv.reset_index(drop=True, inplace=True)
    csv.set_index('time', drop=True, inplace=True)
    csv.drop(['name'], axis=1, inplace=True)
    return csv


def add_derivation(data):
    data.reset_index(inplace=True)
    derivation = holoborodko_diff(data.air_humidity.values, data.time)
    derivation = pd.Series(derivation, name='air_hum_dt')
    df = derivation.to_frame()
    data = pd.merge(data, df, how='inner', left_index=True, right_index=True)
    data = data[['time', 'pressure', 'air_temp', 'air_humidity', 'air_hum_dt',
                 '69886_rssi', '69886_snr', 'soil_humidity']]
    data.set_index('time', drop=True, inplace=True)
    return data