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