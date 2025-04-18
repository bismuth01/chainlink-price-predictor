# Price Prediction using RNN, LSTM and GRU and discoPOP like loss function discovery on Chainlink data

## Installation
First clone the repository using
`git clone https://github.com/bismuth01/chainlink-price-predictor.git`

Now, your python installation can use `python` or `python3`. Use the appropriate one in the upcoming commands.

(Optional but Recommended) Make a virtual environment using
`python -m venv ./venv`
and activate it using
`source ./venv/bin/activate`

Install the required pip packages using
`pip install -r requirements.txt`

> ⚠️ The files use Tensorflow 2.19.0, a lower version might create issues.
> It is a known issue in Tensorflow 2.x that sometimes tensorflow.keras is not resolved. For that you can use `pip install tf-keras`

## How to setup
Copy the `.env-local` file and rename it to `.env`.
A few values are setup but it's better to set your own.

You need a node url of a node on the ethereum mainnet which you can get from ![chainlist](https://chainlist.org/chain/1) and set it in `NODE_URL`

I have used the `ETH/USD` chainlink proxy contract.
You can find another proxy price feed contract from their ![list of data feed contracts](https://docs.chain.link/data-feeds/price-feeds/addresses?page=1) and set it in `PROXY_CONTRACT_ADDRESS`.

Set a .csv file name for `DATA_OUTPUT_FILE` to get the output of the chainlink historical data in that particular file.

Get a `GEMINI_API_KEY` from Google AI Studio.
Set a .json file name for `LOG_FILE` to log the conversation and statistics with the LLM.

## How to use
First run `chainlink_data.py` which will load the historical data available in the proxy contract. If it is the first time, then you might need to wait a while for it to fetch everything. The file is made such that it resumes where it left, so if fetching fails for some reason, running it again, will make it continue from where it left. Which also means that it can always update to the latest data.

The files are used as follows: -
- `price_predictor_rnn.py` -> Contains RNN model
- `price_predictor_lstm.py` -> Contains LSTM model
- `price_predictor_gru.py` -> Contains GRU model
- `price_predictor_all.py` -> Contains all models, helps in comparision
- `discopop_all.py` -> Chats with a google provided LLM to discover new loss functions similar to ![discoPOP](https://arxiv.org/abs/2406.08414)

Before running any file, at the starting of the file, you can set: -
- `TIME_FRAME` which preprocesses the data to a certain timeframe, which is set to `5min` by default
- `NEXT_PREDICTION_IN_MINUTES` which sets after which timeframe should the upcoming price be predicted, should be set to the same as `TIME_FRME`.
- `SEQ_LEN` which set the length of sequence, i.e., how far back data the model can see at a time, which is set to `20` by default.

## Quickstart resources
`eth_usd_chainlink.csv` file already contains some historical price data on `ETH/USD` price feed which can save some time because it only needs to be updated.
`sample_chat_log.json` file already contains a few conversations with the LLM to find better results fast.
