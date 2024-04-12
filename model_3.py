import keras
import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import pyplot as plt

df = pd.read_csv('1h_last3.csv')

print('\n', df)

print('\n', df.info())

print('\n', df.describe())

titles = ['Open', 'High', 'Low', 'Close', 'Volume']

feature_keys = ['Open', 'High', 'Low', 'Last', 'Volume']

date_time_key = "Time"

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 5


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


print(
    'The selected parameters are:',
    ', '.join([titles[i] for i in [0, 1, 2, 3, 4]])
)
selected_features = [feature_keys[i] for i in [0, 1, 2, 3, 4]]
features = df[selected_features]
features.index = df[date_time_key]
print(features.head(), '\n')

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
print(features.head(), '\n')

train_data = features.loc[0:train_split - 1]
val_data = features.loc[train_split:]

print('train data: ', train_data)
print('val_data: ', val_data)

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(5)]].values
y_train = features.iloc[start:end][[0, 1, 2, 3]]

sequence_length = int(past / step)
print(sequence_length)

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

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(5)]].values
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
lstm_out = keras.layers.LSTM(32)(inputs)
# lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False))(inputs)

# temp = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(inputs)
# lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(10))(temp)

outputs = keras.layers.Dense(4)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
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


def denormalize(norm_data, data_mean, data_std):
    return (norm_data * data_std) + data_mean


# Среднее и стандартное отклонение тренировочных данных
data_mean = train_data.mean(axis=0).values  # Получаем значения как numpy массив
data_std = train_data.std(axis=0).values    # Получаем значения как numpy массив

# Применение обратной нормализации к предсказаниям
original_predictions = denormalize(predictions, data_mean[:4], data_std[:4])

print("Original scale predictions:", original_predictions)


# # creating plot for last 3880 samples
# df['Time'] = pd.to_datetime(df['Time'])
#
# last_3880_df = df.tail(24)
#
# ohlc_df = last_3880_df[['Time', 'Open', 'High', 'Low', 'Last']].copy()
#
# ohlc_df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
#
# ohlc_df.set_index('Date', inplace=True)
#
# mpf.plot(ohlc_df, type='candle', datetime_format='%m/%d/%Y %H:%M',
#          title='true', ylabel='price', style='charles')
#
# mpf.show()
