import keras
import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import pyplot as plt

df = pd.read_csv('1h_last3.csv')

titles = ['Open', 'High', 'Low', 'Close']

feature_keys = ['Open', 'High', 'Low', 'Last']

date_time_key = "Time"

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 2

past = 1000
future = 72
learning_rate = 0.001
batch_size = 128
epochs = 1


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


print(
    'The selected parameters are:',
    ', '.join([titles[i] for i in [0, 1, 2, 3]])
)
selected_features = [feature_keys[i] for i in [0, 1, 2, 3]]
features = df[selected_features]
features.index = df[date_time_key]

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
# print('features', features)

train_data = features.loc[0:train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(4)]].values
y_train = features.iloc[start:end][[0, 1, 2, 3]]

sequence_length = int(past / step)
print('sequence_length', sequence_length)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

x_end = len(val_data) - past - future
print('x_end: ', x_end)

label_start = train_split + past + future
print('label_start: ', label_start)

x_val = val_data.iloc[:x_end][[i for i in range(4)]].values
y_val = features.iloc[label_start:][[0, 1, 2, 3]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

print('Input shape:', inputs.numpy().shape)
print('Target shape:', targets.numpy().shape)

print((inputs.shape[1], inputs.shape[2]))

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
# lstm_out = keras.layers.SimpleRNN(32)(inputs)
lstm_out = keras.layers.LSTM(64)(inputs)
# lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False))(inputs)

# temp = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(inputs)
# lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(10))(temp)

outputs = keras.layers.Dense(4)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.weights.h5"
path_checkpointt = "modell_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpointt,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

# loading best weight
model.load_weights(path_checkpoint)

# predict
predictions = model.predict(dataset_val)

print("Shape of predictions:", predictions.shape)

print("Predictions:", predictions)


def denormalize_predictions(predictions, features, train_split):
    data_mean = features.iloc[:train_split].mean()
    data_std = features.iloc[:train_split].std()
    return predictions * data_std.values + data_mean.values


predictions_denorm = denormalize_predictions(predictions, df[selected_features], train_split)
print("Shape of denormalize predictions:", predictions_denorm.shape)
print("denormalize Predictions:", predictions_denorm)

df_candles = pd.DataFrame(predictions_denorm, columns=['Open', 'High', 'Low', 'Close'])

df_candles.index = pd.date_range(start='17-08-2023 14:00', periods=len(df_candles), freq='h')

print(df_candles.head())

mpf.plot(df_candles, type='candle', style='charles',
         title='predict',
         ylabel='price')

# creating plot for last 3880 samples
df['Time'] = pd.to_datetime(df['Time'])

last_3880_df = df.tail(5386)

ohlc_df = last_3880_df[['Time', 'Open', 'High', 'Low', 'Last']].copy()

ohlc_df.columns = ['Date', 'Open', 'High', 'Low', 'Close']

ohlc_df.set_index('Date', inplace=True)

mpf.plot(ohlc_df, type='candle', datetime_format='%m/%d/%Y %H:%M',
         title='true', ylabel='price', style='charles')

mpf.show()

buffer = features.iloc[-sequence_length:].values
# print('buffer', buffer)
predictions = []

for i in range(24):
    input_buffer = buffer.reshape(1, sequence_length, 4)

    pred = model.predict(input_buffer)

    predictions.append(pred[0])

    buffer = np.vstack([buffer[1:], pred[0]])
    # print(i, buffer)

predictions = np.array(predictions)
# print(predictions)

predictions_df = pd.DataFrame(predictions, columns=['Open', 'High', 'Low', 'Close'])
# print("Predictions for 24h: ", predictions_df)

predictions_denorm = denormalize_predictions(predictions_df, df[selected_features], train_split)

print("predictions denorm for 24h: ", predictions_denorm)

dates = pd.date_range(start="2024-04-02", periods=24, freq='h')
predictions_denorm.index = dates

mpf.plot(predictions_denorm[['Open', 'High', 'Low', 'Close']],
         type='candle', style='charles',
         title='predicted next 24h',
         ylabel='price')

plt.show()

