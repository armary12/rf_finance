import pandas as pd
import sys
import os
from datetime import datetime
from enum import Enum
from stable_baselines3 import DQN
from environment import TradingEnv
import helper as hp
from stable_baselines3.common.env_checker import check_env

class RFEnum(Enum):
    MACD = 'MACD'
    MACD_CROSS = 'MACD_CROSS'
    MULTIPLE_MACD_CROSS = 'multiple_MACD_cross'
    PRICE = 'price'
    RSI = 'RSI'

# Constants and configurations
SEED = 41
# STOCK_NAME = sys.argv[1]
DIRECTORY = 'data/us/2023-2024/1d'
PRETRAINED_MODEL_PATH = 'model/combined_stock_rsi20240814_195233'  # Path to your pre-trained model

MACD_CONFIGS = [{'short_window': 6, 'long_window': 13, 'signal_window': 4}]

# Define features to be used in the environment
feature_baselines = ['Standardized_Adj Close']
feature_baselines.append('Standardized_RSI')
for MACD_CONFIG in MACD_CONFIGS:
    feature_baselines.append(f'Standardized_MACD Histogram_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')
    feature_baselines.append(f'Standardized_MACD_Cross_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')

# Define the reward functions
REWARD = [
    RFEnum.PRICE.value, 
    RFEnum.MACD_CROSS.value,
    RFEnum.RSI.value,
]

# Create the training environment
for index, filename in enumerate(sorted(os.listdir(DIRECTORY))[100:]):
    if filename.endswith('.csv'):
        STOCK_NAME = filename.split('.')[0]

        MODEL_SAVE_DIR = f'model/9_35_500000_0001114_100000_rsi/{STOCK_NAME}'
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

        # Load and preprocess the stock data
        stock_data = pd.read_csv(f'data/us/2020-2022/1d/{STOCK_NAME}.csv')
        stock_data = hp.data_process(stock_data, MACD_CONFIGS)

        env = TradingEnv(stock_data, 
                        features=feature_baselines,
                        window_size=1,
                        reward_function=REWARD,
                        macd_configs=MACD_CONFIGS,
                        stock_name=STOCK_NAME
                        )

        # Load the pre-trained model
        model = DQN.load(PRETRAINED_MODEL_PATH, env=env, seed=SEED, gamma=0.90, exploration_fraction=0.35, buffer_size=500000, learning_rate=0.000114)

        # Fine-tune the model on the specific stock
        model.learn(total_timesteps=100000)

        model.save(f"{MODEL_SAVE_DIR}/{STOCK_NAME}")
        print('index' + str(index))