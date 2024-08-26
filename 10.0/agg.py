import os
import pandas as pd
import glob

# Set the path to the directory containing the folders
base_dir = 'result/transfer_macd'

# Create an empty DataFrame to store the concatenated data
all_data = pd.DataFrame()

# Iterate through each folder in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Ensure we are only processing directories
    if os.path.isdir(folder_path):
        # Find the CSV files that start with 'train' in the current folder
        csv_files = glob.glob(os.path.join(folder_path, 'trade*.csv'))
        
        # Iterate through each found CSV file
        for csv_file in csv_files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)
            
            # Add a column with the stock name (folder name)
            df['Stock'] = folder
            
            # Concatenate the current DataFrame with the previous ones
            all_data = pd.concat([all_data, df], ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
all_data.to_csv('all_stocks_data_macd.csv', index=False)

print("All files have been concatenated into 'all_stocks_data.csv'")
