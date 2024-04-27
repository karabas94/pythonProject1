import keras
import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import pyplot as plt

df = pd.read_csv('final.csv')
print('df', df.shape)

titles = ['Open', 'High', 'Low', 'Close']

feature_keys = ['Open', 'High', 'Low', 'Last']

date_time_key = "Time"

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6

past = 8640
future = 72
learning_rate = 0.001
batch_size = 128
epochs = 100


def normalize(data, data_mean, data_std):
    return (data - data_mean) / data_std


print(
    'The selected parameters are:',
    ', '.join([titles[i] for i in [0, 1, 2, 3]])
)
selected_features = [feature_keys[i] for i in [0, 1, 2, 3]]
features = df[selected_features]
features.index = df[date_time_key]

data_mean = features.iloc[:train_split].mean()
print("data_mean", data_mean)
data_std = features.iloc[:train_split].std()
print("data_std", data_std)

features = normalize(features.values, data_mean.values, data_std.values)
features = pd.DataFrame(features)

train_data = features.loc[0:train_split - 1]
val_data = features.loc[train_split:]
print('train_data', train_data.shape)
print('val_data', val_data.shape)

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

x_val = val_data[[i for i in range(4)]].values
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

path_checkpoint = "5m_model_checkpoint.weights.h5"

es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

# history = model.fit(
#     dataset_train,
#     epochs=epochs,
#     validation_data=dataset_val,
#     callbacks=[es_callback, modelckpt_callback],
# )

# loading best weight
model.load_weights(path_checkpoint)

# -------------predict val-data ----------------------------------------------------------------------------------------
predictions = model.predict(dataset_val)

print("Shape of predictions:", predictions.shape)


def denormalize_predictions(predictions, data_mean, data_std):
    return predictions * data_std + data_mean


predictions_denorm = denormalize_predictions(predictions, data_mean.values, data_std.values)
print("Shape of denormalize predictions:", predictions_denorm.shape)

df_candles = pd.DataFrame(predictions_denorm, columns=['Open', 'High', 'Low', 'Close'])

df_candles.index = pd.date_range(start='23-02-2023 14:35', periods=len(df_candles), freq='5min')

mpf.plot(df_candles, type='candle', style='charles',
         title='predict val data',
         ylabel='price')

# creating plot for last 82497 samples
df['Time'] = pd.to_datetime(df['Time'])

last_3880_df = df.tail(82497)

ohlc_df = last_3880_df[['Time', 'Open', 'High', 'Low', 'Last']].copy()

ohlc_df.columns = ['Date', 'Open', 'High', 'Low', 'Close']

ohlc_df.set_index('Date', inplace=True)

mpf.plot(ohlc_df, type='candle', datetime_format='%m/%d/%Y %H:%M',
         title='true val data', ylabel='price', style='charles')

mpf.show()
# ----------------------preprocess predict data ------------------------------------------------------------------------
df_pred = pd.read_csv('pred5m.csv')
feature_keys_pred = ['Open', 'High', 'Low', 'Last']
selected_features_pred = [feature_keys_pred[i] for i in [0, 1, 2, 3]]
features_pred = df_pred[selected_features_pred]

features_pred = normalize(features_pred.values, data_mean.values, data_std.values)
features_pred = pd.DataFrame(features_pred)
print('features_pred', features_pred)
# ------------------------predict next day------------------------------------------------------------------------------
last_inputs = features_pred[-sequence_length:].values
pred = []

for i in range(288):
    dataset_predict = keras.preprocessing.timeseries_dataset_from_array(
        last_inputs,
        None,
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=1,
    )

    predictions_next = model.predict(dataset_predict)
    predictions_next_denorm = denormalize_predictions(predictions_next, data_mean.values, data_std.values)
    pred.append(predictions_next_denorm[0].flatten())
    last_inputs = np.vstack([last_inputs[1:], predictions_next[0]])

print(pred)

predictions_df = pd.DataFrame(pred, columns=['Open', 'High', 'Low', 'Close'])
dates = pd.date_range(start="2024-04-11 00:00", periods=len(pred), freq='5min')
predictions_df.index = dates

mpf.plot(predictions_df,
         type='candle',
         style='charles',
         title='predicted next day',
         ylabel='price',
         volume=False,
         figsize=(12, 6))


# -----------------------true next day----------------------------------------------------------------------------------
df_true = pd.read_csv('true5m.csv')
df_true['Time'] = pd.to_datetime(df_true['Time'])
df_true.set_index('Time', inplace=True)
df_true.rename(columns={'Last': 'Close'}, inplace=True)
mpf.plot(df_true, type='candle', datetime_format='%m/%d/%Y %H:%M',
         title='true next day', ylabel='price', style='charles')
mpf.show()
