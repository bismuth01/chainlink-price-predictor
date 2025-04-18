from web3 import Web3
import json
import pandas as pd
from datetime import datetime
import os
import csv
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = os.environ['DATA_OUTPUT_FILE']

w3 = Web3(Web3.HTTPProvider(os.environ['NODE_URL']))

# Chainlink ETH/USD Proxy Price Feed contract
price_feed_address = os.environ['PROXY_CONTRACT_ADDRESS']
price_feed_abi = json.loads('[{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]')

# Contract instance
price_feed_contract = w3.eth.contract(address=price_feed_address, abi=price_feed_abi)

def get_latest_eth_price():
    data = price_feed_contract.functions.latestRoundData().call()
    price = data[1] / 10**8  # Price has 8 decimal places
    timestamp = datetime.fromtimestamp(data[3])
    roundId = data[0]
    return {"timestamp": timestamp, "price": price, "roundId": roundId}

def get_historical_eth_price(round : int):
    data = price_feed_contract.functions.getRoundData(round).call()
    price = data[1] / 10**8  # Price has 8 decimal places
    timestamp = datetime.fromtimestamp(data[3])
    round = data[0]
    return {"timestamp": timestamp, "price": price}

exists = os.path.exists(OUTPUT_FILE)
data_file = open(OUTPUT_FILE, 'a', newline='')
csvfile = csv.writer(data_file, delimiter=',')
if not exists:
    csvfile.writerow(['roundId', 'timestamp', 'price'])

if __name__ == '__main__':
    try:
        mask = int('0xFFFFFFFFFFFFFFFF', 16)

        latest_eth_price = get_latest_eth_price()
        latest_roundId = int(latest_eth_price['roundId'])
        latest_round = latest_roundId & mask
        print("Latest roundId: ", latest_round)

        start_roundId = 0
        try:
            with open(OUTPUT_FILE, 'r') as f:
                last_line = f.readlines()[-1]
                last_round = last_line.split(',')[0]
                start_roundId = int(last_round) + 1
        except Exception:
            start_roundId = latest_roundId - latest_round + 1

        print(f"{'Round':<20} | {'Timestamp':<20} | {'Price':<10}")
        for i in range(start_roundId, latest_roundId + 1):
            round_data = get_historical_eth_price(i)
            timestamp = round_data['timestamp']
            price = round_data['price']
            csvfile.writerow([i, timestamp, price])
            print(f"{i:<20} | {str(timestamp):<20} | {price:<10}")

        data_file.close()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        data_file.close()
