import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os

DATE = datetime.today().strftime('%Y%m%d_%H%M%S')

class TradingEnv(gym.Env):
    def __init__(self, stock_data, save_directory_path='', trade_size=10000, features=['Adj Close'], window_size=10, reward_function=['price'], macd_configs=[], stock_name='', seed=42, buy_fee=0.000, sell_fee=0.000):
        super(TradingEnv, self).__init__()

        # Define the directory paths
        self.directory_path = f"{save_directory_path}"
        self.csv_file_path = os.path.join(self.directory_path, f"train_{DATE}.csv")
        self.trade_csv_file_path = os.path.join(self.directory_path, f"trade_{DATE}.csv")
        self.plot_file_name = os.path.join(self.directory_path, DATE)

        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)
        self.stock_name = stock_name
        # Initialize stock data using the 'Adj Close' column
        self.stock_data = stock_data
        self.stock_price = stock_data['Adj Close'].values
        self.current_step = 0
        self.window_size = window_size
        self.features = features
        self.macd_histograms, self.macds, self.macd_signals, self.macd_crosses = self._get_MACD_data(macd_configs)
        self.macd_configs = macd_configs
        self.rsi = stock_data['RSI'].values
        self.reward_function = reward_function
        # Define observation space (stock prices for the last 10 days + cash balance)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size * len(self.features),), dtype=np.float64)

        # Trade size and fees
        self.trade_size = trade_size
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        # Initialize initial state
        self.cash_balance = 10000  # initial cash balance
        self.stock_owned = 0       # initial stocks owned
        self.initial_cash_balance = self.cash_balance

        # Stop-loss variables
        self.last_buy_price = None
        self.last_sell_price = None
        self.stop_loss_threshold = 0.99  # 5% stop-loss for buy, or 5% stop-loss for sell
        self.stop_loss_actions = []  # Track stop-loss actions

        # Track total assets at last buy/sell action for win/loss calculation
        self.last_buy_assets = None
        self.last_sell_assets = None
        
        # For tracking performance
        self.total_assets_history = []
        self.actions = []
        self.actions_map = {}
        self.episode_rewards = []
        self.current_episode_reward = 0

        # For recording actions and rewards
        self.record = []

        # Create directory if not exists
        # os.makedirs(self.directory_path, exist_ok=True)

        # Initialize trade counters
        self.win_trades = 0
        self.loss_trades = 0
        self.total_trades = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # This ensures the seed is set correctly
        self.current_step = 0
        self.cash_balance = 10000
        self.stock_owned = 0
        self.total_assets_history = []
        self.actions = []
        self.actions_map = {}
        self.current_episode_reward = 0
        self.record = []
        self.last_buy_price = None  # Reset last buy price on reset
        self.last_sell_price = None  # Reset last sell price on reset
        self.last_buy_assets = None  # Reset last buy assets on reset
        self.last_sell_assets = None  # Reset last sell assets on reset
        self.stop_loss_actions = []  # Reset stop-loss actions on reset
        self.win_trades = 0  # Reset win trades
        self.loss_trades = 0  # Reset loss trades
        self.total_trades = 0  # Reset total trades
        return self._get_observation(), {}

    def _get_observation(self):
        window_data = self.stock_data[self.features][self.current_step:self.current_step + self.window_size]
        return np.array(window_data).flatten()
    
    def step(self, action):
        current_price = self.stock_price[self.current_step + self.window_size]
        current_rsi = self.rsi[self.current_step + self.window_size]
        current_macd_histogram = self.macd_histograms[0][self.current_step + self.window_size]
        current_macd_cross = self.macd_crosses[0][self.current_step + self.window_size]

        if action == 1:  # Buy
            num_stocks_to_buy = min(self.trade_size, self.cash_balance // current_price)
            if num_stocks_to_buy > 0:
                total_cost = num_stocks_to_buy * current_price * (1 + self.buy_fee)
                self.stock_owned += num_stocks_to_buy
                self.cash_balance -= total_cost
                self.last_buy_price = current_price  # Update last buy price
                self.last_buy_assets = self.cash_balance + self.stock_owned * current_price
                self.total_trades += 1  # Increment total trades
                # Calculate if previous sell was a win or loss
                if self.last_sell_assets is not None:
                    if self.last_sell_assets < self.last_buy_assets:
                        self.win_trades += 1
                    else:
                        self.loss_trades += 1
            else:
                action = 0

        elif action == 2:  # Sell
            if self.stock_owned > 0:
                num_stocks_to_sell = min(self.trade_size, self.stock_owned)
                total_revenue = num_stocks_to_sell * current_price * (1 - self.sell_fee)
                self.stock_owned -= num_stocks_to_sell
                self.cash_balance += total_revenue
                self.last_sell_price = current_price  # Update last sell price
                self.last_sell_assets = self.cash_balance + self.stock_owned * current_price
                self.total_trades += 1  # Increment total trades
                # Calculate if previous buy was a win or loss
                if self.last_buy_assets is not None:
                    if self.last_buy_assets < self.last_sell_assets:
                        self.win_trades += 1
                    else:
                        self.loss_trades += 1
            else:
                action = 0

        # Calculate reward
        next_price = self.stock_price[self.current_step + self.window_size + 1]
        total_asset_before = self.cash_balance + self.stock_owned * current_price
        total_asset_after = self.cash_balance + self.stock_owned * next_price

        reward = 0
        reward_components = {}

        if 'MACD' in self.reward_function:
            MACD_reward = self._get_MACD_reward(action, current_macd_histogram)
            reward_components['MACD_reward'] = MACD_reward
            reward += MACD_reward
        if 'multiple_MACD_cross' in self.reward_function:
            MACD_reward = self._get_multiple_MACD_cross_reward()
            reward_components['multiple_MACD_cross_reward'] = MACD_reward
            reward += MACD_reward
        if 'RSI' in self.reward_function:
            RSI_reward = self._get_RSI_reward(action, current_rsi)
            reward_components['RSI_reward'] = RSI_reward
            reward += RSI_reward
        if 'MACD_CROSS' in self.reward_function:
            MACD_CROSS_reward = self._get_MACD_CROSS_reward(action, current_macd_cross)
            reward_components['MACD_CROSS_reward'] = MACD_CROSS_reward
            reward += MACD_CROSS_reward
        if 'price' in self.reward_function:
            price_reward = np.log(total_asset_after / total_asset_before)
            reward_components['price_reward'] = price_reward
            reward += price_reward

        self.current_episode_reward += reward

        self.current_step += 1
        done = self.current_step + self.window_size + 1 >= len(self.stock_data)

        # Track total assets
        total_assets = self.cash_balance + self.stock_owned * current_price
        self.total_assets_history.append(total_assets)

        # Track actions for plotting
        self.actions.append(action)
        self.actions_map[self.current_step + self.window_size] = action

        # Evaluate the final action at the end of the episode
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            last_action = self._find_last_non_zero_action()
            if last_action == 1 and self.last_sell_assets is not None:
                # If the last action was a buy, check the previous sell
                if self.last_sell_assets < total_assets:
                    self.win_trades += 1
                else:
                    self.loss_trades += 1
            elif last_action == 2 and self.last_buy_assets is not None:
                # If the last action was a sell, check the previous buy
                if self.last_buy_assets < total_assets:
                    self.win_trades += 1
                else:
                    self.loss_trades += 1

        # Record action and rewards
        self.record.append({
            'date': self.stock_data.iloc[self.current_step + self.window_size]['Date'],
            'action': action,
            'price_reward': reward_components.get('price_reward', 0),
            'MACD_reward': reward_components.get('MACD_reward', 0),
            'multiple_MACD_cross_reward': reward_components.get('multiple_MACD_cross_reward', 0),
            'RSI_reward': reward_components.get('RSI_reward', 0),
            'MACD_cross_reward': reward_components.get('MACD_CROSS_reward', 0),
            'current_MACD_cross_0': self.macd_crosses[0][self.current_step + self.window_size],
            'current_price': current_price,
            'current_MACD_histrogram': current_macd_histogram,
            'total_reward': reward,
            'cash_balance': self.cash_balance,
            'stock_owned': self.stock_owned,
            'total_assets': total_assets
        })

        return self._get_observation(), reward, done, False, {}

    def _find_last_non_zero_action(self):
        # Iterate over the actions in reverse
        for action in reversed(self.actions):
            if action != 0:
                return action
        return None  # Return None if no non-zero action is found

    def render(self, mode='human', close=False):
        current_price = self.stock_price[self.current_step + self.window_size]
        total_assets = self.cash_balance + self.stock_owned * current_price
        profit = total_assets - 10000
        print(f'Step: {self.current_step}')
        print(f'Cash Balance: {self.cash_balance}')
        print(f'Stocks Owned: {self.stock_owned}')
        print(f'Total Assets: {total_assets}')
        print(f'Profit: {profit}')
        print(f"Winning Trades: {self.win_trades}")
        print(f"Losing Trades: {self.loss_trades}")
        print(f"Total Trades: {self.total_trades}")

    def _get_MACD_data(self, macd_configs):
        macd_histograms = []
        macds = []
        macd_signals = []
        macd_cross = []

        for macd_config in macd_configs:
            short = macd_config['short_window']
            long = macd_config['long_window']
            signal = macd_config['signal_window']

            macds.append(self.stock_data['MACD_{}_{}'.format(short, long)].values)
            macd_histograms.append(self.stock_data['MACD Histogram_{}_{}_{}'.format(short, long, signal)].values)
            macd_signals.append(self.stock_data['Signal_{}_{}_{}'.format(short, long, signal)])
            macd_cross.append(self.stock_data['MACD_Cross_{}_{}_{}'.format(short, long, signal)])
        return macd_histograms, macds, macd_signals, macd_cross

    def _get_multiple_MACD_cross_reward(self):
        multiple_macd_cross_reward = 0
        
        if self.actions_map.get(self.current_step - 2, 0) == 0:
            return 0

        confirm_cross = False
        if self.actions_map.get(self.current_step - 2, 0) == 1:
            for i in range(self.window_size -1 ):
                if self.macd_crosses[1][self.current_step - i] > 0:
                    confirm_cross = True
        elif self.actions_map.get(self.current_step - 2, 0) == 2:
            for i in range(self.window_size -1 ):
                if self.macd_crosses[1][self.current_step - i] < 0:
                    confirm_cross = True
        
        if confirm_cross:
            multiple_macd_cross_reward += 0.05
        else:
            multiple_macd_cross_reward -= 0.05

        return multiple_macd_cross_reward

    def _get_MACD_CROSS_reward(self, action, macd_cross):
        macd_reward = 0
        if (action == 1 and macd_cross > 0):  # Reward buying when MACD Cross up 
            macd_reward += 0.01
        elif (action == 2 and macd_cross < 0):  # Reward selling when MACD Cross down
            macd_reward += 0.01
        elif (action == 1 and macd_cross <= 0):  # Penalize buying when MACD Cross down
            macd_reward -= 0.01
        elif (action == 2 and macd_cross >= 0):  # Penalize selling when MACD Cross up
            macd_reward -= 0.01
        return macd_reward

    def _get_MACD_reward(self, action, macd_histogram):
        macd_reward = 0
        if (action == 1 and macd_histogram > 0):  # Reward buying when MACD histogram is positive
            macd_reward += 0.01 * macd_histogram
        elif (action == 2 and macd_histogram < 0):  # Reward selling when MACD histogram is negative
            macd_reward += 0.01 * abs(macd_histogram)
        return macd_reward

    def _get_RSI_reward(self, action, current_rsi):
        rsi_reward = 0
        if (action == 1 and current_rsi < 30):  # Reward buying at low RSI
            rsi_reward += 0.01 * (30 - current_rsi)
        elif (action == 2 and current_rsi > 70):  # Reward selling at high RSI
            rsi_reward += 0.01 * (current_rsi - 70)
        return rsi_reward

    def plot_performance(self):
        # Extract start and end dates
        start_date = self.stock_data.iloc[0]['Date']
        end_date = self.stock_data.iloc[-1]['Date']

        num_macd_configs = len(self.macd_histograms)
        plt.figure(figsize=(12, 6 + 3 * num_macd_configs))

        # Plot 1: Stock Price with Buy and Sell Signals
        plt.subplot(num_macd_configs + 2, 1, 1)
        stock_prices = self.stock_price[self.window_size:self.current_step + self.window_size + 1]
        plt.plot(stock_prices, label='Stock Price')

        # Mark buy (1) and sell (2) actions
        buy_actions = [i for i, a in enumerate(self.actions) if a == 1]
        sell_actions = [i for i, a in enumerate(self.actions) if a == 2]
        plt.plot(buy_actions, [stock_prices[i] for i in buy_actions], '^', markersize=10, color='g', label='Buy')
        plt.plot(sell_actions, [stock_prices[i] for i in sell_actions], 'v', markersize=10, color='r', label='Sell')

        # Mark stop-loss actions
        plt.plot(self.stop_loss_actions, [stock_prices[i] for i in self.stop_loss_actions], 'x', markersize=10, color='b', label='Stop Loss')

        plt.xlabel('Step')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price with Buy, Sell, and Stop Loss Signals for {self.stock_name} from {start_date} to {end_date}')
        plt.legend()

        # Plot MACD Histograms
        for idx in range(num_macd_configs):
            short = self.macd_configs[idx]['short_window']
            long = self.macd_configs[idx]['long_window']
            signal_n = self.macd_configs[idx]['signal_window']

            plt.subplot(num_macd_configs + 2, 1, idx + 2)
            macd_histogram = self.macd_histograms[idx][self.window_size:self.current_step + self.window_size + 1]
            macd = self.macds[idx][self.window_size:self.current_step + self.window_size + 1]
            signal = self.macd_signals[idx][self.window_size:self.current_step + self.window_size + 1]
            plt.bar(range(len(macd_histogram)), macd_histogram, label='MACD Histogram', color='b')
            plt.plot(macd, label='MACD', color='g')
            plt.plot(signal, label='Signal Line', color='r')
            plt.legend(loc='upper left')
            plt.title(f'MACD Histogram {short} {long} {signal_n}')
            plt.xlabel('Step')
            plt.ylabel('MACD Histogram')
            plt.xticks(rotation=45)
            plt.grid()

        # Plot Total Assets and Buy and Hold Strategy Over Time
        plt.subplot(num_macd_configs + 2, 1, num_macd_configs + 2)
        plt.plot(self.total_assets_history, label='Total Assets', color='r')

        # Calculate buy-and-hold strategy performance
        initial_price = self.stock_price[self.window_size]
        buy_and_hold_assets = [(self.initial_cash_balance / initial_price) * p for p in stock_prices]
        plt.plot(buy_and_hold_assets, label='Buy and Hold Strategy', color='b')

        # MACD Trading Strategy
        macd_trading_assets = self.simulate_macd_strategy()
        plt.plot(macd_trading_assets, label='MACD Strategy', color='g')

        plt.xlabel('Step')
        plt.ylabel('Assets')

        plt.title(f'Total Assets')
        plt.legend()

        plt.tight_layout()
        
        os.makedirs(self.directory_path, exist_ok=True)       
        plt.savefig(self.plot_file_name)

        # plt.show()

    def simulate_macd_strategy(self):
        cash_balance = self.initial_cash_balance
        stock_owned = 0
        macd_trading_assets = []

        for i in range(self.window_size, self.current_step + self.window_size + 1):
            action = macd_trading_strategy(self.macd_histograms[0], i)
            current_price = self.stock_price[i]

            if action == 1:  # Buy
                num_stocks_to_buy = min(self.trade_size, cash_balance // current_price)
                if num_stocks_to_buy > 0:
                    stock_owned += num_stocks_to_buy
                    cash_balance -= num_stocks_to_buy * current_price * (1 + self.buy_fee)
            elif action == 2:  # Sell
                if stock_owned > 0:
                    num_stocks_to_sell = min(self.trade_size, stock_owned)
                    stock_owned -= num_stocks_to_sell
                    cash_balance += num_stocks_to_sell * current_price * (1 - self.sell_fee)

            total_assets = cash_balance + stock_owned * current_price
            macd_trading_assets.append(total_assets)

        return macd_trading_assets

    def plot_episode_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards Over Time')
        plt.legend()
        plt.show()

    def get_total_assets_history(self):
        return self.total_assets_history

    def get_trade_summary(self):
        return {
            'win_trades': self.win_trades,
            'loss_trades': self.loss_trades,
            'total_trades': self.total_trades
        }

    def print_trade_summary(self):
        print(f"Winning Trades: {self.win_trades}")
        print(f"Losing Trades: {self.loss_trades}")
        print(f"Total Trades: {self.total_trades}")

    def get_env_information(self):
        return 'Reward function: ' + self.reward_function + ' | Window size: ' + str(self.window_size)

    def save_attributes_to_text(self, model_path, data_path):
        attributes = {
            'trade_size': self.trade_size,
            'features': self.features,
            'window_size': self.window_size,
            'reward_function': self.reward_function,
            'macd_configs': self.macd_configs,
            'stock_name': self.stock_name,
            'cash_balance': self.cash_balance,
            'initial_cash_balance': self.initial_cash_balance,
            'buy_fee': self.buy_fee,
            'sell_fee': self.sell_fee,
            'win_trades': self.win_trades,
            'loss_trades': self.loss_trades,
            'total_trades': self.total_trades,
            'model_path': model_path,
            'data_path' : data_path
        }
        
        attributes_file_path = os.path.join(self.directory_path, f"attributes_{DATE}.txt")
        with open(attributes_file_path, 'w') as file:
            for key, value in attributes.items():
                file.write(f"{key}: {value}\n")
        print(f"Attributes saved to {attributes_file_path}")

    def save_record_to_csv(self):
        df = pd.DataFrame(self.record)
        df.to_csv(self.csv_file_path, index=False)
        print(f"Record saved to {self.csv_file_path}")

    def save_trade_summary_to_csv(self):
        # Prepare data to be saved
        trade_summary = {
            'Agent_Assets': [self.get_total_assets_history()[-1]],
            'Buy_and_Hold_Assets': [self._calculate_buy_and_hold_assets()],
            'Total_Trades': [self.total_trades],
            'Winning_Trades': [self.win_trades],
            'Losing_Trades': [self.loss_trades]
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(trade_summary)

        # Save to CSV
        df.to_csv(self.trade_csv_file_path, index=False)
        print(f"Trade summary saved to {self.trade_csv_file_path}")

    def _calculate_buy_and_hold_assets(self):
        # Calculate buy-and-hold strategy performance
        stock_prices = self.stock_price[self.window_size:self.current_step + self.window_size + 1]
        initial_price = self.stock_price[self.window_size]
        buy_and_hold_assets = (self.initial_cash_balance / initial_price) * stock_prices[-1]
        return buy_and_hold_assets

def macd_trading_strategy(macd_histogram, current_step):
    if current_step == 1:
        if macd_histogram[1] > 0:
            return 1
        else:
            return 0

    prev_histogram = macd_histogram[current_step - 1]
    curr_histogram = macd_histogram[current_step]

    if prev_histogram < 0 and curr_histogram > 0:
        return 1  # Buy
    elif prev_histogram > 0 and curr_histogram < 0:
        return 2  # Sell
    else:
        return 0
