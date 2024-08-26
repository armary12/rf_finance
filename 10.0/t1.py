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
STOCK_NAME = sys.argv[1]
MODEL_PATH = 'model/stock_' + datetime.today().strftime('%Y%m%d_%H%M%S')
MACD_CONFIGS = [{'short_window':6, 'long_window':13, 'signal_window': 4}]

stock_data = pd.read_csv(f'data/us/2020-2021/1d/{STOCK_NAME}.csv')
stock_data = hp.data_process(stock_data, MACD_CONFIGS)

feature_baselines = ['Standardized_Adj Close']
feature_baselines.append('Standardized_RSI')
for MACD_CONFIG in MACD_CONFIGS:
    feature_baselines.append(f'Standardized_MACD Histogram_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')
    feature_baselines.append(f'Standardized_MACD_Cross_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')


REWARD = [
          RFEnum.PRICE.value, 
          RFEnum.MACD_CROSS.value,
          RFEnum.RSI.value
        ]

# Create the environment
env = TradingEnv(stock_data, 
                 features=feature_baselines,
                 window_size=1,
                 reward_function=REWARD,
                 macd_configs=MACD_CONFIGS,
                 stock_name=STOCK_NAME
                 )

model = DQN('MlpPolicy', env=env, verbose=1, gamma=0.99, exploration_fraction=0.26, buffer_size=940886, learning_rate=0.000214, seed=SEED)
model.learn(total_timesteps=100000)
model.save(MODEL_PATH)

stock_data_test = pd.read_csv(f'data/us/2022-2023/1d/{STOCK_NAME}.csv')
stock_data_test = hp.data_process(stock_data_test, MACD_CONFIGS)

env_test = TradingEnv(stock_data_test, 
                 features=feature_baselines,
                 window_size=1,
                 reward_function=REWARD,
                 macd_configs=MACD_CONFIGS,
                 stock_name=STOCK_NAME)

obs, info = env_test.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, truncate, info = env_test.step(action)
    if done:
        break

print(env_test.get_total_assets_history()[-1])
env_test.plot_performance()
env_test.save_record_to_csv()
