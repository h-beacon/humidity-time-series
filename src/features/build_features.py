import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

#POVEZATI SA make_dataset.py
def process_data(data, step):
    values = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    split = int(len(data) * 0.8)
    data_train = scaled[:split]
    data_test = scaled[split:]
    x_train, y_train = split_sequences(data_train, step)
    x_test, y_test = split_sequences(data_test, step)
    return x_train, y_train, x_test, y_test, scaler