import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def round_minutes(tm):
    tm = tm - timedelta(minutes=tm.minute % 10, seconds=tm.second)
    return tm


def merge_data(csv_1, csv_2, csv_3):
    merge = pd.merge(csv_1, csv_2, how='inner', left_index=True, right_index=True)
    merged = pd.merge(merge, csv_3, how='inner', left_index=True, right_index=True)
    return merged


def load_raw_data(soil_path, pressure_path, air_path):
    soil = pd.read_csv(soil_path)
    pressure = pd.read_csv(pressure_path)
    air = pd.read_csv(air_path)
    return soil, pressure, air

