# %% Import required libraries and functions
import pandas as pd
import numpy as np
from datetime import datetime
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

import matplotlib.pyplot as plt

SEQ_LEN = 20 # Number of days of sequence the ML model gets to look into
TIME_FRAME = '5min' # Interval made in data
NEXT_PREDICTION_IN_MINUTES = 5

def create_sequences(data, seq_len, label_index):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X_data = data.iloc[i:(i+seq_len)].values
        y_data = data.iloc[i+ seq_len][label_index]
        X.append(X_data)
        y.append(y_data)

    return np.array(X), np.array(y)

def add_technical_indicators(df, label_index):
    df['sma5'] = SMAIndicator(close=df[label_index], window=5).sma_indicator()
    df['sma10'] = SMAIndicator(close=df[label_index], window=10).sma_indicator()
    df['sma20'] = SMAIndicator(close=df[label_index], window=20).sma_indicator()

    df['ema_9'] = EMAIndicator(close=df[label_index], window=9).ema_indicator()
    df['ema_21'] = EMAIndicator(close=df[label_index], window=21).ema_indicator()

    macd = MACD(close=df[label_index])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    bollinger = BollingerBands(close=df[label_index])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    df['atr'] = AverageTrueRange(high=df[label_index], low=df[label_index], close=df[label_index]).average_true_range()

    stoch = StochasticOscillator(high=df[label_index], low=df[label_index], close=df[label_index])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['rsi_7'] = RSIIndicator(close=df[label_index], window=7).rsi()
    df['rsi_14'] = RSIIndicator(close=df[label_index], window=14).rsi()

    return df

def draw_corr(df, y_index):
    corr_matrix = df.corr()

    print(corr_matrix[y_index])

def predict_next_price(df, model, seq_len, scaler, y_scaler):
    latest_data = df.iloc[-seq_len:].copy()

    X_new = latest_data.values.reshape(1, seq_len, latest_data.shape[1])
    X_new_reshaped = X_new.reshape(seq_len, latest_data.shape[1])
    X_new_scaled = scaler.transform(X_new_reshaped)
    X_new_scaled = X_new_scaled.reshape(1, seq_len, latest_data.shape[1])

    pred_scaled = model.predict(X_new_scaled)

    next_price = y_scaler.inverse_transform(pred_scaled)[0][0]

    return next_price

# %% Importing the dataset
df = pd.read_csv('eth_usd_chainlink.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['timestamp', 'roundId'])
df = df.drop_duplicates(subset='timestamp', keep='last')

df = df.set_index('timestamp')
print(df.describe())

# %% Creating per 5min data
start_time = df.index.min().floor(TIME_FRAME)
end_time = df.index.max().ceil(TIME_FRAME)

regular_grid = pd.date_range(start=start_time, end=end_time, freq=TIME_FRAME)

df_regular = df.reindex(df.index.union(regular_grid)).sort_index()
df_regular['price'] = df_regular['price'].interpolate(method='time')
df_5min = df_regular.loc[regular_grid]

df_processed = pd.DataFrame(df_5min['price'])

df_processed = add_technical_indicators(df_processed, 'price')
df_processed = df_processed.dropna()

draw_corr(df_processed, 'price')

X, y = create_sequences(df_processed, SEQ_LEN, 'price')

# %% Min Max Scaling the data
n_samples = X.shape[0]
n_steps = X.shape[1]
n_features = X.shape[2]

X_reshaped = X.reshape(n_samples * n_steps, n_features)

scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X_reshaped)

X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)

y_scaler = MinMaxScaler(feature_range=(0,1))
y_scaled = y_scaler.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, shuffle=False)

# %% Creating the RNN model
rnn_model = Sequential([
    Input(shape=(n_steps, n_features)),
    SimpleRNN(128, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(128),
    Dropout(0.2),
    Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mse')

rnn_model.summary()

# %% Creating the LSTM model
lstm_model = Sequential([
    Input(shape=(n_steps, n_features)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

# %% Creating the GRU model
gru_model = Sequential([
    Input(shape=(n_steps, n_features)),
    GRU(128, return_sequences=True),
    Dropout(0.2),
    GRU(128),
    Dropout(0.2),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mse')

gru_model.summary()

# %% Training the RNN model
earlystopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, start_from_epoch=30)
rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[earlystopping]
)

# %% Training the LSTM model
earlystopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, start_from_epoch=30)
lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[earlystopping]
)

# %% Training the GRU model
earlystopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, start_from_epoch=30)
gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[earlystopping]
)

# %% Plotting the loss
plt.figure(figsize=(12,6))
plt.plot(rnn_history.history['loss'], label='RNN Train loss')
plt.plot(rnn_history.history['val_loss'], label='RNN Validation Loss')
plt.plot(lstm_history.history['loss'], label='LSTM Train loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.plot(gru_history.history['loss'], label='GRU Train loss')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% Predicting test data and generating metrics
rnn_y_pred_scaled = rnn_model.predict(X_test)
lstm_y_pred_scaled = lstm_model.predict(X_test)
gru_y_pred_scaled = gru_model.predict(X_test)

rnn_y_pred = y_scaler.inverse_transform(rnn_y_pred_scaled)
lstm_y_pred = y_scaler.inverse_transform(lstm_y_pred_scaled)
gru_y_pred = y_scaler.inverse_transform(gru_y_pred_scaled)

y_test_actual = y_scaler.inverse_transform(y_test)

rnn_rmse = math.sqrt(mean_squared_error(y_test_actual, rnn_y_pred))
lstm_rmse = math.sqrt(mean_squared_error(y_test_actual, lstm_y_pred))
gru_rmse = math.sqrt(mean_squared_error(y_test_actual, gru_y_pred))

rnn_mae = mean_absolute_error(y_test_actual, rnn_y_pred)
lstm_mae = mean_absolute_error(y_test_actual, lstm_y_pred)
gru_mae = mean_absolute_error(y_test_actual, gru_y_pred)

rnn_r2 = r2_score(y_test_actual, rnn_y_pred)
lstm_r2 = r2_score(y_test_actual, lstm_y_pred)
gru_r2 = r2_score(y_test_actual, gru_y_pred)

print("Root Mean Squared Error: -")
print(f"RNN : {rnn_rmse}\nLSTM: {lstm_rmse}\nGRU: {gru_rmse}\n")

print("Mean Absolute Error: -")
print(f"RNN : {rnn_mae}\nLSTM: {lstm_mae}\nGRU: {gru_mae}\n")

print("R2 Score: -")
print(f"RNN : {rnn_r2}\nLSTM: {lstm_r2}\nGRU: {gru_r2}\n")

# %% Plotting actual and predicted price
plt.figure(figsize=(15, 8))
plt.plot(y_test_actual, label='Actual Close Price')
plt.plot(rnn_y_pred, label='RNN Predicted Close Price')
plt.plot(lstm_y_pred, label='LSTM Predicted Close Price')
plt.plot(gru_y_pred, label='GRU Predicted Close Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# %% Predicting the next price from last available data
last_timestamp = df_processed.index[-1]
next_timestamp = last_timestamp + pd.Timedelta(minutes=NEXT_PREDICTION_IN_MINUTES)

rnn_next_price = predict_next_price(df_processed, rnn_model, SEQ_LEN, scaler, y_scaler)
lstm_next_price = predict_next_price(df_processed, lstm_model, SEQ_LEN, scaler, y_scaler)
gru_next_price = predict_next_price(df_processed, gru_model, SEQ_LEN, scaler, y_scaler)

print(f"Predicted price at {next_timestamp}: -")
print(f"RNN: ${rnn_next_price:.2f}\nLSTM: ${lstm_next_price:.2f}\nGRU: ${gru_next_price:.2f}\n")

# %% Visualizing where the predicted price lies
plt.figure(figsize=(15, 8))
plt.plot(df_processed.index[-100:], df_processed['price'][-100:], label='Historical Price')
plt.scatter([next_timestamp], [rnn_next_price], color='red', s=30, label='RNN Predicted Next Price')
plt.scatter([next_timestamp], [lstm_next_price], color='green', s=30, label='LSTM Predicted Next Price')
plt.scatter([next_timestamp], [gru_next_price], color='blue', s=30, label='GRU Predicted Next Price')
plt.title('ETH/USD Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
