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
STOCK_PATH = sys.argv[1]
MODEL_PATH = None
TESTING = False
if len(sys.argv) == 3:
    MODEL_PATH = sys.argv[2]
    TESTING = True
else:
    MODEL_PATH = 'model/stock_' + datetime.today().strftime('%Y%m%d_%H%M%S')
    
STOCK_NAME = STOCK_PATH.split('/')[-1].split('.')[0]

MACD_CONFIGS = [{'short_window':6, 'long_window':13, 'signal_window': 4}]

stock_data = pd.read_csv(STOCK_PATH)
stock_data = hp.data_process(stock_data, MACD_CONFIGS)
stock_data.to_csv('check')

feature_baselines = ['Standardized_Adj Close']
feature_baselines.append('Standardized_RSI')
for MACD_CONFIG in MACD_CONFIGS:
    feature_baselines.append(f'Standardized_MACD Histogram_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')
    feature_baselines.append(f'Standardized_MACD_Cross_{MACD_CONFIG["short_window"]}_{MACD_CONFIG["long_window"]}_{MACD_CONFIG["signal_window"]}')

# Create the environment
env = TradingEnv(stock_data,
                 'result/test',
                 features=feature_baselines,
                 window_size=1,
                 reward_function=[RFEnum.PRICE.value,
                                  RFEnum.MACD_CROSS.value,
                                  RFEnum.RSI.value
                                  ],
                 macd_configs=MACD_CONFIGS,
                 stock_name=STOCK_NAME,
                 )

# Check the environment

if not TESTING:
    model = DQN('MlpPolicy', env=env, verbose=1, gamma=0.99, exploration_fraction=0.26, buffer_size=940886, learning_rate=0.000214, seed=SEED)
    model.learn(total_timesteps=100000)
    model.save(MODEL_PATH)
else:
    model = DQN.load(MODEL_PATH, env= env, seed=SEED, gamma=0.95, exploration_fraction=0.35, buffer_size=500000, learning_rate=0.000114)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, truncate, info = env.step(action)
    if done:
        break

# Plot performance
print(env.get_total_assets_history()[-1])
env.plot_performance()
# env.save_record_to_csv()