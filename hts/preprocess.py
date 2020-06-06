import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

def process_data(data, test_data, step, ratio):
    values = data.values
    test_values = test_data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    test_scaled = scaler.transform(test_values)
    split = int(len(data) * ratio)
    test_split = int(len(test_data) * ratio)
    data_train = scaled[:split]
    data_valid = scaled[split:]
    data_test = test_scaled[test_split:]
    x_train, y_train = split_sequences(data_train, step)
    x_valid, y_valid = split_sequences(data_valid, step)
    x_test, y_test = split_sequences(data_test, step)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, scaler

def clean(csv,  roll_step=None, temp=False, absolute=False):
    """Returns the cleaned version of raw csv file."""
    csv['f3c80_time'].fillna(csv['69886_time'])
    csv.drop(['name','time','69886_time','ae05e_rssi','ae05e_snr','ae05e_time'], axis=1, inplace=True)
    csv.rename(columns={'f3c80_time': 'Time'}, inplace=True)
    csv['Time'] = csv.Time.astype(str)
    csv['Time'] = csv['Time'].map(lambda x: x.lstrip(',.').rstrip('0123456789Z').strip(',.'))
    csv['Time'] = pd.to_datetime(csv['Time'], format="%Y-%m-%d %H")
    csv['degreesC'] = csv['degreesC'].map(lambda x: (x/10)-2)
    csv['humidity'] = csv['humidity'].map(lambda x: (x-220)/11)
    if temp is False:
        csv.drop(['degreesC'], axis=1, inplace=True)
    csv.drop(csv[(csv.humidity > 100) | (csv.humidity < 5)].index, inplace=True)
    csv.drop(csv[csv.Time.dt.year < 2019].index, inplace=True)
    csv.dropna(axis=0, inplace=True)
    csv['Time'] = csv['Time'].dt.floor('Min')
    csv = csv.drop_duplicates('Time', keep='last')
    if temp is True:
        csv = csv[['Time','69886_rssi','f3c80_rssi','69886_snr','f3c80_snr','degreesC','humidity']]
    else:
        csv = csv[['Time','69886_rssi','f3c80_rssi','69886_snr','f3c80_snr','humidity']]
    csv.reset_index(drop=True, inplace=True)
    csv.set_index('Time', drop=True, inplace=True)
    # deep sensor
    if len(csv.index) > 31500:
        csv.drop(csv.loc['2020-01-07':'2020-01-31'].index, axis=0, inplace=True)
        if roll_step is not None:
            for col in ['69886_rssi', 'f3c80_rssi', '69886_snr', 'f3c80_snr']:
                csv[col] = csv[col].rolling(roll_step, min_periods=1).median()
    # shallow sensor
    elif len(csv.index) < 31500:
        csv.drop(csv.loc['2019-12-23':'2020-01-21'].index, axis=0, inplace=True)
        if roll_step is not None:
            for col in ['69886_rssi', 'f3c80_rssi', '69886_snr', 'f3c80_snr']:
                csv[col] = csv[col].rolling(roll_step, min_periods=1).median()
    if absolute is True:
        for var in ['69886_rssi','f3c80_rssi','69886_snr','f3c80_snr']:
            csv[var] = csv[var].map(lambda x: np.power(10, x/10))
    return csv