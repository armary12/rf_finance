import os
from environment import TradingEnv
import pandas as pd
import helper as hp
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import sys
from enum import Enum
from datetime import datetime

class RFEnum(Enum):
    MACD = 'MACD'
    MACD_CROSS = 'MACD_CROSS'
    MULTIPLE_MACD_CROSS = 'multiple_MACD_cross'
    PRICE = 'price'
    RSI = 'RSI'

SEED = 41
STOCK_FOLDER = sys.argv[1]
MODEL_PATH = 'model/combined_stock_rsi' + datetime.today().strftime('%Y%m%d_%H%M%S')

MACD_CONFIGS = [{'short_window': 6, 'long_window': 13, 'signal_window': 4}]

# Initialize the model as None before starting the loop
model = None

# Loop through each file in the folder and train the model sequentially on each stock
for  index, filename in enumerate(os.listdir(STOCK_FOLDER)):
    if filename.endswith('.csv'):
        STOCK_NAME = filename.split('.')[0]
        stock_data = pd.read_csv(os.path.join(STOCK_FOLDER, filename))
        stock_data = hp.data_process(stock_data, MACD_CONFIGS)

        feature_baselines = ['Standardized_Adj Close']
        # feature_baselines.append('Standardized_RSI')
        for MACD_CONFIG in MACD_CONFIGS:
            feature_baselines.append(f'Standardized_MACD Histogram_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')
            feature_baselines.append(f'Standardized_MACD_Cross_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')

        REWARD = [
            RFEnum.PRICE.value,
            RFEnum.MACD_CROSS.value,
            # RFEnum.RSI.value
        ]

        # Create the environment for the current stock
        env = TradingEnv(stock_data,
                         features=feature_baselines,
                         window_size=1,
                         reward_function=REWARD,
                         macd_configs=MACD_CONFIGS,
                         stock_name=STOCK_NAME
                         )

        if model is None:
            # Initialize the model for the first time
            model = DQN('MlpPolicy', env=env, verbose=1, gamma=0.95, exploration_fraction=0.26, buffer_size=940886, learning_rate=0.000214, seed=SEED)
        else:
            # Reset the environment for continued training
            model.set_env(env)

        # Train the model on the current stock data
        model.learn(total_timesteps=100000)
        print(index)

# Save the model after training on all stocks
model.save(MODEL_PATH)
print(f"Model trained on all stocks sequentially and saved at {MODEL_PATH}")
