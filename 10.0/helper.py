from sklearn.preprocessing import StandardScaler
import pandas as pd

def add_macd_rsi(df, column='Adj Close', macd_configs = [{'short_window':12, 'long_window':26, 'signal_window': 9}]):
    
    for macd_config in macd_configs:
        short = macd_config['short_window']
        long = macd_config['long_window']
        signal = macd_config['signal_window']
        
        ema_short_text = 'EMA_' + str(short)
        ema_long_text = 'EMA_' + str(long)
        macd_text = 'MACD_' + str(short) + '_' + str(long)
        signal_text = 'Signal_' + str(short) + '_' + str(long) + '_' + str(signal)
        macd_his_text = 'MACD Histogram_' + str(short) + '_' + str(long) + '_' + str(signal)
        macd_cross_text = 'MACD_Cross_' + str(short) + '_' + str(long) + '_' + str(signal)

        df[ema_short_text] = df[column].ewm(span=short, adjust=False).mean()
        df[ema_long_text] = df[column].ewm(span=long, adjust=False).mean()
        df[macd_text] = df[ema_short_text] - df[ema_long_text]
        df[signal_text] = df[macd_text].ewm(span=signal, adjust=False).mean()
        df[macd_his_text] = df[macd_text] - df[signal_text]
        
        # Calculate MACD cross
        df[macd_cross_text] = 0
        df.loc[df[macd_text] > df[signal_text], macd_cross_text] = 1
        df.loc[df[macd_text] < df[signal_text], macd_cross_text] = -1
        df[macd_cross_text] = df[macd_cross_text].diff().fillna(0)
        
        # Drop the intermediate columns used for calculations
        df.drop([ema_short_text, ema_long_text], axis=1, inplace=True)

    # Calculate RSI
    window_length = 14
    
    # Get the difference in price from previous step
    delta = df[column].diff()
    
    # Make the positive gains (up) and negative gains (down) Series
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calculate the EWMA
    avg_gain = gain.ewm(com=(window_length - 1), min_periods=window_length).mean()
    avg_loss = loss.ewm(com=(window_length - 1), min_periods=window_length).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
        
    # Drop the first 26 rows to account for the long_window period
    df = df.iloc[14:].reset_index(drop=True)

    return df

def data_process(data, macd_configs):
    data['pct_change'] = data['Adj Close'].pct_change()
    data = add_macd_rsi(data, column='Adj Close', macd_configs=macd_configs)

    columns_to_scale = [col for col in data.columns if col != 'Date']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns_to_scale])
    
    scaled_data_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=data.index)
    
    # Add standardized columns as new columns
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:  # Ensure we only standardize numeric columns
            data[f"Standardized_{column}"] = scaled_data_df[column]

    return data
