import os
from environment import TradingEnv
import pandas as pd
import helper as hp
from stable_baselines3 import DQN
import sys
from enum import Enum

class RFEnum(Enum):
    MACD = 'MACD'
    MACD_CROSS = 'MACD_CROSS'
    MULTIPLE_MACD_CROSS = 'multiple_MACD_cross'
    PRICE = 'price'
    RSI = 'RSI'

SEED = 41
DIRECTORY = 'data/us/2023-2024/1d'
TESTING = True

for filename in os.listdir(DIRECTORY):
    STOCK_PATH = os.path.join(DIRECTORY, filename)
    STOCK_NAME = filename.split('.')[0]
    MODEL_PATH = f'model/9_35_500000_0001114_50000_macd/{STOCK_NAME}/{STOCK_NAME}.zip'

    DIRECTORY_PATH = f"result/transfer_macd/{STOCK_NAME}"

    MACD_CONFIGS = [{'short_window':6, 'long_window':13, 'signal_window': 4}]

    stock_data = pd.read_csv(STOCK_PATH)
    stock_data = hp.data_process(stock_data, MACD_CONFIGS)

    feature_baselines = ['Standardized_Adj Close']
    # feature_baselines.append('Standardized_RSI')
    for MACD_CONFIG in MACD_CONFIGS:
        feature_baselines.append(f'Standardized_MACD Histogram_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')
        feature_baselines.append(f'Standardized_MACD_Cross_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')

    # Create the environment
    env = TradingEnv(stock_data,
                        DIRECTORY_PATH,
                        features=feature_baselines,
                        window_size=1,
                        reward_function=[RFEnum.PRICE.value,
                                        RFEnum.MACD_CROSS.value,
                                        # RFEnum.RSI.value
                                        ],
                        macd_configs=MACD_CONFIGS,
                        stock_name=STOCK_NAME,
                        )

    # Load the model
    model = DQN.load(MODEL_PATH, env= env, seed=SEED)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, truncate, info = env.step(action)
        if done:
            break

    # Plot performance and save results
    print(f"{STOCK_NAME}: Final Asset Value = {env.get_total_assets_history()[-1]}")
    env.plot_performance()
    env.save_record_to_csv()
    env.save_attributes_to_text(MODEL_PATH, STOCK_PATH)
    env.save_trade_summary_to_csv()
