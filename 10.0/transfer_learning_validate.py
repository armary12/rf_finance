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
STOCK_NAME = sys.argv[1]
PRETRAINED_MODEL_PATH = 'model/combined_stock_20240814_194650.zip'  # Path to your pre-trained model
MODEL_SAVE_DIR = f'model/{STOCK_NAME}'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

MACD_CONFIGS = [{'short_window': 6, 'long_window': 13, 'signal_window': 4}]

# Load and preprocess the stock data
stock_data = pd.read_csv(f'data/us/2020-2022/1d/{STOCK_NAME}.csv')
stock_data = hp.data_process(stock_data, MACD_CONFIGS)

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
    RFEnum.RSI.value
]

# Create the training environment
env = TradingEnv(stock_data, 
                 features=feature_baselines,
                 window_size=1,
                 reward_function=REWARD,
                 macd_configs=MACD_CONFIGS,
                 stock_name=STOCK_NAME
                 )

# Load and preprocess the validation data
validation_data_path = f'data/us/2023-2024/1d/{STOCK_NAME}.csv'
validation_data = pd.read_csv(validation_data_path)
validation_data = hp.data_process(validation_data, MACD_CONFIGS)

# Create the validation environment
validation_env = TradingEnv(validation_data, 
                            features=feature_baselines,
                            window_size=1,
                            reward_function=REWARD,
                            macd_configs=MACD_CONFIGS,
                            stock_name=STOCK_NAME
                            )
# check_env(validation_env)
# Hyperparameter tuning configurations
train_steps_list = [20000, 50000]
gamma_list = [0.90, 0.95, 0.99]

best_total_return = -float('inf')
best_model_path = None

# Hyperparameter tuning loop
for train_steps in train_steps_list:
    for gamma in gamma_list:
        print(f"Training with total_timesteps={train_steps} and gamma={gamma}")

        # Load the pre-trained model
        model = DQN.load(PRETRAINED_MODEL_PATH, env=env, seed=SEED, gamma=gamma, exploration_fraction=0.35, buffer_size=500000, learning_rate=0.000114)

        # Fine-tune the model on the specific stock
        model.learn(total_timesteps=train_steps)

        # Evaluate the model on the validation environment
        obs, info = validation_env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, truncate, info = validation_env.step(action)
            
        total_return = env.get_total_assets_history()[-1]
        print(f"Total rewards on validation data: {str(env.get_total_assets_history()[-1])}")

        # Save the best model
        if total_return > best_total_return:
            best_total_return = total_return
            best_model_path = f'{MODEL_SAVE_DIR}/best_model_timesteps_{train_steps}_gamma_{gamma}_{datetime.today().strftime("%Y%m%d_%H%M%S")}.zip'
            model.save(best_model_path)
            print(f"New best model saved: {best_model_path}")

print(f"Best model saved at: {best_model_path} with total rewards: {best_total_return}")
