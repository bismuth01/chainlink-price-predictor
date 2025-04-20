# Chainlink Price Predictor

## 🔍 Introduction
This project leverages neural networks like RNN, LSTM, and GRU to predict Ethereum (ETH) prices using historical data fetched directly from Chainlink oracles. Inspired by the [**discoPOP**](https://arxiv.org/abs/2406.08414) paper, it includes an automated framework for discovering custom loss functions using Google's Gemini LLM to better suit the volatility and dynamics of cryptocurrency markets.

With dynamic loss experimentation, time-frame flexibility, and reproducible setup, this project is designed for research, financial modeling, and exploring LLM-assisted ML workflows.

## 📦 Installation
First clone the repository:
```bash
git clone https://github.com/bismuth01/chainlink-price-predictor.git
cd chainlink-price-predictor
```

Now, your python installation can use `python` or `python3`. Use the appropriate one in the upcoming commands.

(Optional but Recommended) Make a virtual environment:
```bash
python -m venv ./venv
source ./venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

> ⚠️ Note: This project uses TensorFlow 2.19.0. Using an older version might cause issues.
> If you encounter import errors like tensorflow.keras not being resolved, try:
> `pip install tf-keras`

## ⚙️ Setup
1. Copy the .env-local file and rename it to .env.
2. Edit .env with your own values:
- NODE_URL: Ethereum mainnet RPC node URL (get one from [Chainlist](https://chainlist.org/chain/1))
- PROXY_CONTRACT_ADDRESS: Chainlink ETH/USD price feed proxy or your custom feed (see [list](https://docs.chain.link/data-feeds/price-feeds/addresses?page=1))
- DATA_OUTPUT_FILE: Output .csv file to store fetched historical data
- GEMINI_API_KEY: API key from Google AI Studio
- LOG_FILE: Output .json file to store LLM interaction logs

## 🚀 How to use

### Fetch Historical Price Data
```bash
python chainlink_data.py
```
- On the first run, it fetches all historical data.
- The script resumes from where it left off if interrupted.
- You can rerun it anytime to update the data.

### Using the models
The files are used as follows: -
- `price_predictor_rnn.py` -> Contains RNN model
- `price_predictor_lstm.py` -> Contains LSTM model
- `price_predictor_gru.py` -> Contains GRU model
- `price_predictor_all.py` -> Contains all models, helps in comparision

#### Configurable Parameters
At the top of each script, you can modify:
- `TIME_FRAME`: Timeframe for downsampling data (default: `5min`)
- `NEXT_PREDICTION_IN_MINUTES`: Time ahead to predict (should match `TIME_FRAME`)
- `SEQ_LEN`: Number of past data points the model can see (default: `20`)
- `NUM_NEXT_PREDICTION_POINTS`: Number of past predictions to plot with the future price prediction (default: `5`)

### Using custom loss function generation
`discopop_all.py` -> This interacts with the Gemini LLM to generate custom loss functions, inspired by [discoPOP](https://arxiv.org/abs/2406.08414), tailored for financial time series prediction.

After each response from the LLM, the custom loss function is extracted and used in RNN, LSTM and GRU model training. The results are sent back to the LLM to help make better functions.
It records the previous conversations and saves it to the set `LOG_FILE` so history is remembered by the Gemini LLM when re-run to produce better results.

#### Configurable Parameters
At the top of the script, you can modify:
- `FUNCTION_EPOCHS`: Number of times functions to generate and evaluate (default: `2`)

## ⚡ Quickstart resources
`eth_usd_chainlink.csv`: Pre-fetched ETH/USD price data to skip initial fetching
`sample_chat_log.json`: Example Gemini conversation logs to speed up loss function discovery
