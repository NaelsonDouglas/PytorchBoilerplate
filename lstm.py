import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def performance(gain:float) -> float:
    MIN_GAIN = 1.0125
    (LOSS, LATERAL, GAIN) = (-1, 0, 1)
    if gain >= MIN_GAIN:
        return GAIN
    elif gain <= 1.0:
        return LOSS
    else:
        return LATERAL

def calc_metrics(df:pd.DataFrame) -> pd.DataFrame:
    df = df[['open_time', 'open', 'close', 'volume']]
    df['volumevolume'] = df['volume'].shift(1)/df['volume'].shift(2)
    df['previous_gain'] = df['close']/df['open']
    df['previous_gain'] = df['previous_gain'].shift(1)
    df['result'] = df['previous_gain'].apply(performance)
    return df

def add_btc_gain(other:pd.DataFrame, btc:pd.DataFrame) -> pd.DataFrame:
    other['previous_btc_gain'] = btc['previous_gain'].apply(lambda x: round(x, 5))
    return other

def train_test_val(df:pd.DataFrame) -> tuple:
    train, test_val = train_test_split(df, train_size=0.8, shuffle=False)
    test, val = train_test_split(test_val, train_size=0.5, shuffle=False)
    return (train, test, val)

def make_sample(window:np.array) -> np.array:
    window = window[0][0]
    nof_features = window.shape[-1]
    features = window[:,0:nof_features-1].T
    target = window[:,-1]

def get_windows(df:pd.DataFrame, window_size:int):
    min_class = df['result'].value_counts().min()
    classes = df['result'].value_counts().reset_index()
    classes = {class_value:nof_samples for (class_value,nof_samples) in  classes.values}
    nof_features = len(df.columns)
    sliding_windows = sliding_window_view(df.values, (window_size, nof_features))
    sliding_windows = sliding_windows.reshape(len(sliding_windows), window_size, nof_features)
    result = list()
    for (class_value, nof_samples) in classes.items():
        samples = np.array([w for w in sliding_windows if w[-1,-1] == class_value])
        idx = np.random.randint(len(samples), size=min_class)
        samples = samples[idx,:]
        result.append(samples)
    result = np.vstack(result)
    result = [_format_window(w) for w in result]
    return result

def _format_window(window:np.array) -> np.array:
    (_, nof_columns) = window.shape
    nof_features = nof_columns-1
    target = window[-1,-1]
    win = list(window[:,0:nof_features].T)
    win.append(np.array(target))
    return win


if __name__ == '__main__':
    ltc = pd.read_parquet('ltc.parquet')
    ltc = calc_metrics(ltc)

    btc = pd.read_parquet('btc.parquet')
    btc = calc_metrics(btc)

    ltc = add_btc_gain(ltc, btc)

    ltc = ltc[['open_time', 'open', 'volume', 'volumevolume', 'previous_gain', 'previous_btc_gain', 'result']]
    ltc = ltc[['volumevolume', 'previous_gain', 'previous_btc_gain', 'result']]
    ltc = ltc[['previous_gain', 'result']]
    ltc = ltc.dropna()
    windows = get_windows(ltc, 24)
