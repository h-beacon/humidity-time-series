import numpy as np
import pandas as pd

def clean(csv,  roll_step, temp=False, absolute=False):
    """ Input je csv file, ako je temp false izbacuje temperaturu zemlje,
        ako je absolute True prebacuje u apsolutnu skalu"""

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
    # Deeper sensor
    if len(csv.index) > 18000:
        csv.drop(csv.loc['2020-01-07':'2020-01-31'].index, axis=0, inplace=True)
        for col in ['69886_rssi', 'f3c80_rssi', '69886_snr', 'f3c80_snr']:
            csv[col] = csv[col].rolling(roll_step, min_periods=1).median()
    # Shallow sensor
    elif (len(csv.index) < 18000) and (len(csv.index) > 17000):
        csv.drop(csv.loc['2019-12-23':'2020-01-21'].index, axis=0, inplace=True)
        for col in ['69886_rssi', 'f3c80_rssi', '69886_snr', 'f3c80_snr']:
            csv[col] = csv[col].rolling(roll_step, min_periods=1).median()
    if absolute is True:
        for var in ['69886_rssi','f3c80_rssi','69886_snr','f3c80_snr']:
            csv[var] = csv[var].map(lambda x: np.power(10, x/10))
    return csv

