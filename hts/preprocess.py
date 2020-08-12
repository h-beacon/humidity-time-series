import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from .utils import round_minutes, holoborodko_diff, cumulative
import json
import warnings
warnings.filterwarnings("ignore")


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


def split_data(data, ratio, test_data=None):
    split = int(len(data) * ratio)
    if ratio <= 0.8:
        test_split = int(len(data) * (ratio + 0.1))
    elif ratio == 0.9:
        test_split = int(len(data) * (ratio + 0.05))
    values = data.values
    if test_data is None:
        train = values[:split]
        valid = values[split:test_split]
        test = values[test_split:]
    else:
        test_values = test_data.values
        test_data_split = int(len(test_data) * ratio)
        train = values
        valid = test_values[:test_data_split]
        test = test_values[test_data_split:]
    return train, valid, test


def process_data_rnn(data, step, ratio, test_data=None):
    train, valid, test = split_data(data, ratio, test_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train = scaler.fit_transform(train)
    data_valid = scaler.transform(valid)
    data_test = scaler.transform(test)
    x_train, y_train = split_sequences(data_train, step)
    x_valid, y_valid = split_sequences(data_valid, step)
    x_test, y_test = split_sequences(data_test, step)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, scaler


def parse_json_data(csv):
    rssi_69886 = []
    rssi_f3c80 = []
    snr_69886 = []
    snr_f3c80 = []
    humidity = []
    temp = []
    df = csv[['name', 'time']]
    for row in range(0, len(csv)):
        a = csv.payload_fields[row]
        b = csv.gateways[row]
        if isinstance(a, str):
            pf = json.loads(a)
            humidity.append(pf['humidity'])
            temp.append(pf['tempearture'])
        else:
            humidity.append(np.nan)
            temp.append(np.nan)
        gw = json.loads(b)
        if not any(d['gtw_id'] == 'eui-b827ebfffe069886' for d in gw):
            rssi_69886.append(np.nan)
            snr_69886.append(np.nan)
        if not any(d['gtw_id'] == 'eui-b827ebfffebf3c80' for d in gw):
            rssi_f3c80.append(np.nan)
            snr_f3c80.append(np.nan)
        gen = (item for item in gw)
        for i in gen:
            if i['gtw_id'] == 'eui-b827ebfffe069886':
                rssi_69886.append(i['rssi'])
                snr_69886.append(i['snr'])
            elif i['gtw_id'] == 'eui-b827ebfffebf3c80':
                rssi_f3c80.append(i['rssi'])
                snr_f3c80.append(i['snr'])
    df['69886_rssi'] = rssi_69886
    df['f3c80_rssi'] = rssi_f3c80
    df['69886_snr'] = snr_69886
    df['f3c80_snr'] = snr_f3c80
    df['degreesC'] = temp
    df['humidity'] = humidity
    return df


def clean_soil(csv, absolute=False):
    if "Senzor_zemlje_2" in csv.name.values:
        csv = csv[['name', 'time', '69886_rssi', '69886_snr', 'degreesC', 'humidity']]
    elif "Senzor_zemlje" in csv.name.values:
        csv = csv[['name', 'time', '69886_rssi', 'f3c80_rssi', '69886_snr', 'f3c80_snr', 'degreesC', 'humidity']]
    elif "sensor_earth" in csv.name.values:
        csv = csv[['name', 'time', '69886_rssi', '69886_snr', 'degreesC', 'humidity']]
    csv.time = csv.time.astype(str)
    csv['time'] = csv['time'].map(lambda x: x[0:10])
    csv.time = csv.time.astype(int)
    csv['time'] = csv['time'].map(lambda x: datetime.utcfromtimestamp(x))
    if ("Senzor_zemlje_2" or "Senzor_zemlje") in csv.name.values:
        csv['degreesC'] = csv['degreesC'].map(lambda x: (x/10)-2)
        csv['humidity'] = csv['humidity'].map(lambda x: (x-220)/11)
    csv.drop(csv[(csv.humidity > 100) | (csv.humidity < 5)].index, inplace=True)
    csv.drop(csv[csv.time.dt.year < 2019].index, inplace=True)
    csv.rename(columns={'degreesC': 'soil_temp', 'humidity': 'soil_humidity'}, inplace=True)
    csv['time'] = csv['time'].dt.floor('Min')
    csv = csv.drop_duplicates('time', keep='last')
    if "Senzor_zemlje_2" in csv.name.values:
        csv.drop(csv.loc['2020-01-07':'2020-01-31'].index, axis=0, inplace=True)
    elif "Senzor_zemlje" in csv.name.values:
        csv.drop(csv.loc['2019-12-23':'2020-01-21'].index, axis=0, inplace=True)
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
    if "Senzor_zemlje_2" in csv.name.values:
        csv.drop(csv.loc['2020-07-01':'2020-07-04'].index, axis=0, inplace=True)
    elif "Senzor_zemlje" in csv.name.values:
        csv.drop(csv.loc['2020-07-01':].index, axis=0, inplace=True)
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


def add_derivation(data, column):
    new_column = column + '_dt'
    data.reset_index(inplace=True)
    derivation = holoborodko_diff(data[column].values, data.time)
    derivation = pd.Series(derivation, name=new_column)
    df = derivation.to_frame()
    data = pd.merge(data, df, how='inner', left_index=True, right_index=True)
    data.set_index('time', drop=True, inplace=True)
    return data


def additional_processing(csv):
    #cols = ['pressure', 'air_temp', 'air_humidity', '69886_rssi', '69886_snr', 'soil_humidity']
    cols = ['69886_rssi', '69886_snr', 'soil_humidity']
    for col in cols:
        csv = add_derivation(csv, col)
    csv = csv[['pressure', 'air_temp', 'air_humidity', '69886_rssi', '69886_snr',
               '69886_rssi_dt', '69886_snr_dt', 'soil_humidity_dt']]
    csv = cumulative(csv)
    return csv


def process_data_mlp(data, ratio, test_data=None):
    train, valid, test = split_data(data, ratio, test_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    valid = scaler.transform(valid)
    test = scaler.transform(test)
    x_train, y_train = train[:, :-1], train[:, -1]
    x_valid, y_valid = valid[:, :-1], valid[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    return x_train, y_train, x_valid, y_valid, x_test, y_test, scaler


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg