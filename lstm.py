import pandas as pd
import seaborn as sns
# import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import math
# from tensorflow import keras
# from tensorflow.keras import layers

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def performance(gain:float) -> float:
    MIN_GAIN = 1.0275
    (LOSS, LATERAL, GAIN) = (-1, 0, 1)
    if gain >= MIN_GAIN:
        return GAIN
    elif gain <= 1.0:
        return LOSS
    else:
        return LATERAL


# class Lstm(keras.Sequential):
#     def __init__(self):
#         super().__init__()
#         self.add(layers.LSTM(64, input_shape=(None, 28)))
#         self.add(layers.BatchNormalization())
#         self.add(layers.Dense(10))
#         self.compile(
#             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             optimizer='sgd',
#             metrics=['accuracy'],
#         )

def calc_metrics(df:pd.DataFrame) -> pd.DataFrame:
    df = df[['date', 'open', 'close', 'volume_24h']]
    df['open-1'] = df['open'].shift(1)
    df['volume-1'] = df['volume_24h'].shift(1)
    df['openopen'] = df['open']/df['open-1']
    df['volumevolume'] = df['volume_24h']/df['volume-1']
    df['gain'] = df['close']/df['open']
    df['result'] = df['gain'].apply(performance)
    return df

def get_df(source_file:str, other_id:int) -> pd.DataFrame:
    df = pd.read_csv(source_file)
    other = calc_metrics(df.loc[df['coin_id'] == other_id])
    btc = calc_metrics(df.loc[df['coin_id'] == 1])
    btc = btc.loc[btc.date.isin(other.date)]
    other['btc_gain'] = btc['gain'].apply(lambda x: round(x, 5))
    return other

def train_test_val(df:pd.DataFrame) -> tuple:
    train, test_val = train_test_split(df, train_size=0.8, shuffle=False)
    test, val = train_test_split(test_val, train_size=0.5, shuffle=False)
    return (train, test, val)

if __name__ == '__main__':
    df = get_df('nano_btc.csv',2)
    # df = df.loc[df['result'].isin([-1, 1])]
    # df = df.sample(1500)
    df['openopen'] = df['openopen'].apply(lambda x: (1-x)**3)
    df['volumevolume'] = df['volumevolume'].apply(lambda x: math.tanh(x)**2.2)
    sns.scatterplot(df, x='openopen', y='volumevolume', hue='result')
    plt.show()
    # model = Lstm()
