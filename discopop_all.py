import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import re
import numpy as np
import pandas as pd
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

load_dotenv()

FUNCTION_EPOCHS = 2
LOG_FILE = os.environ['LOG_FILE']

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
sys_msg = """
You are a machine learning researcher focused on discovering novel loss functions to improve stock price prediction. Your models (RNN, LSTM, GRU) are already compiled in TensorFlow and trained using sequences of historical stock prices.
The model is only given the input of the close prices of a few intervals to give a prediction.
When you respond, output a JSON, which has proper escape character to be directly parsed by the json package, where:

- "thought" is your reasoning behind the design of the new loss function (you can reference known time-series or regression loss designs from literature like quantile loss, Huber loss, etc.).

- "name" is a short descriptive name for your new loss function.

- "code" is the exact TensorFlow 2.19.0 compatible Python function implementing this loss function. All models using this function expect an input of in the form of a tuple ( n_steps, n_features) where n_steps and n_features are pre-defined variables. Make sure that the loss function cannot return a NaN loss.

The model code and the calculation of metrics are
```python
rnn_model = Sequential([
    Input(shape=(n_steps, n_features)),
    SimpleRNN(128, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(128),
    Dropout(0.2),
    Dense(1)
])

lstm_model = Sequential([
    Input(shape=(n_steps, n_features)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(1)
])

gru_model = Sequential([
    Input(shape=(n_steps, n_features)),
    GRU(128, return_sequences=True),
    Dropout(0.2),
    GRU(128),
    Dropout(0.2),
    Dense(1)
])

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
```

The loss function must follow this interface:
```python
def custom_loss(y_true, y_pred):
    # your code here
    return loss
```

You can define constants (like thresholds) inside the function, but do not rely on external variables or classes. Be creative and use your knowledge of time-series forecasting, volatility modeling, and robust regression.

You are deeply familiar with:

- Traditional losses like MSE, MAE, Huber.

- Financial modeling considerations like penalizing volatility misprediction, tail risks, or direction accuracy.

- Research from probabilistic forecasting, heteroscedastic regression, and robust statistics.

Return only a JSON like the following:
```json
{
  "thought": "I want to penalize the model more when it predicts the wrong direction of price movement even if the absolute error is small. I'll combine MSE with a directional penalty term.",
  "name": "directional_mse",
  "code": "def custom_loss(y_true, y_pred):
      import tensorflow as tf
      direction_true = tf.sign(y_true[:, 1:] - y_true[:, :-1])
      direction_pred = tf.sign(y_pred[:, 1:] - y_pred[:, :-1])
      direction_penalty = tf.reduce_mean(tf.cast(tf.not_equal(direction_true, direction_pred), tf.float32))
      mse = tf.reduce_mean(tf.square(y_true - y_pred))
      return mse + 0.5 * direction_penalty"
}
```

Your goal is to discover creative, practical loss functions that lead to improved generalization and robustness in stock price prediction models.
"""

def escape_code_block(response_text):
    # Find the code block and escape its inner newlines and quotes properly
    json_pattern = r'```(?:json)?\s*({.*?})\s*```'
    json_match = re.search(json_pattern, response_text, re.DOTALL)

    if json_match:
        return json_match.group(1)

    # If no code block, try to extract just the JSON
    json_pattern = r'{[\s\S]*}'
    json_match = re.search(json_pattern, response_text, re.DOTALL)

    if json_match:
        return json_match.group(0)

    return response_text

def create_history(file_name):
    history = []
    chat_log = []
    with open(file_name, 'r') as file:
        chat_history = json.load(file)
        for convo in chat_history:
            history.append(types.Content(role=convo['role'], parts=[types.Part(text = convo['response'])]))
            chat_log.append(convo)
    return history, chat_log

def custom_loss():
    pass

SEQ_LEN = 20 #Number of days of sequence the ML model gets to look into

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

df = pd.read_csv('eth_usd_chainlink.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['timestamp', 'roundId'])
df = df.drop_duplicates(subset='timestamp', keep='last')

df = df.set_index('timestamp')
print(df.describe())

# %% Creating per 5min data
start_time = df.index.min().floor('5min')
end_time = df.index.max().ceil('5min')

regular_grid = pd.date_range(start=start_time, end=end_time, freq='5min')

df_regular = df.reindex(df.index.union(regular_grid)).sort_index()
df_regular['price'] = df_regular['price'].interpolate(method='time')
df_5min = df_regular.loc[regular_grid]

df_processed = pd.DataFrame(df_5min['price'])

df_processed = add_technical_indicators(df_processed, 'price')
df_processed = df_processed.dropna()

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

rnn_model = Sequential([
    Input(shape=(n_steps, n_features)),
    SimpleRNN(128, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(128),
    Dropout(0.2),
    Dense(1)
])

lstm_model = Sequential([
    Input(shape=(n_steps, n_features)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(1)
])

gru_model = Sequential([
    Input(shape=(n_steps, n_features)),
    GRU(128, return_sequences=True),
    Dropout(0.2),
    GRU(128),
    Dropout(0.2),
    Dense(1)
])

if __name__ == '__main__':
    chat_log = []

    try:
        with open(LOG_FILE, 'r') as f:
            prev_log = f.read()
    except FileNotFoundError:
        print('Log File not found, creating one')
        with open(LOG_FILE, 'w') as f:
            json.dump([], f, indent=4)

    try:
        history, chat_log = create_history(LOG_FILE)
        chat = client.chats.create(model='gemini-2.0-flash', config=types.GenerateContentConfig(
            system_instruction=sys_msg,
            ),
            history=history)

        user_input = 'Please generate the next one'
        chat_log.append({'role': 'user', 'response' : user_input})
        for _ in range(FUNCTION_EPOCHS):
            response = chat.send_message(user_input)
            user_input = 'Please generate the next one'

            if response.text is not None:
                uncode_json = '\n'.join(response.text.split('\n')[1:-1]) # Removing the code blocks
                escaped_json_data = escape_code_block(uncode_json)
                print(escaped_json_data)
                parsed_data = json.loads(escaped_json_data)

                chat_log.append({'role': 'model', 'response' : escaped_json_data})
                exec(parsed_data['code'])

                try:
                    rnn_model.compile(optimizer='adam', loss=custom_loss)
                    rnn_model.summary()

                    lstm_model.compile(optimizer='adam', loss=custom_loss)
                    lstm_model.summary()

                    gru_model.compile(optimizer='adam', loss=custom_loss)
                    gru_model.summary()

                    earlystopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, start_from_epoch=30)
                    rnn_history = rnn_model.fit(
                        X_train,
                        y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[earlystopping]
                    )

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

                    user_input = f"Root Mean Squared Error: -\nRNN : {rnn_rmse}\nLSTM: {lstm_rmse}\nGRU: {gru_rmse}\n" + user_input
                    user_input = f"Mean Absolute Error: -\nRNN : {rnn_mae}\nLSTM: {lstm_mae}\nGRU: {gru_mae}\n" + user_input
                    user_input = f"R2 Score: -\nRNN : {rnn_r2}\nLSTM: {lstm_r2}\nGRU: {gru_r2}\n" + user_input

                    chat_log.append({'role': 'user', 'response' : user_input})

                except Exception:
                    print('Custom function failed')
                    user_input = 'Custom function failed'
            else:
                print('No response found')
                chat_log.append({'role': 'model', 'response' : ''})

    except KeyboardInterrupt:
        print('Keyboard Interrupt caught')

    finally:
        with open(LOG_FILE, 'w') as log_file:
            json.dump(chat_log, log_file, indent=4)
        print('Chat log saved')
