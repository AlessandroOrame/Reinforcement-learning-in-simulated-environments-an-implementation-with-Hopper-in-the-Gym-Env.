import pandas as pd
import matplotlib.pyplot as plt
import glob

file_paths = glob.glob('log/*.csv')

dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path, names=['Episode', 'Return'], skiprows=1)
    dataframes.append(df)

plt.figure(figsize=(10, 6))

for i, df in enumerate(dataframes):
    df['Return_moving_avg'] = df['Return'].rolling(window=100).mean()
    if i == 0:
        plt.plot(df['Episode'], df['Return_moving_avg'], label='REINFORCE')
    elif i == 1:
        plt.plot(df['Episode'], df['Return_moving_avg'], label=f'Actor-Critic')
    elif i == 2:
        plt.plot(df['Episode'], df['Return_moving_avg'], label=f'REINFORCE (b=20)')
    else:
        plt.plot(df['Episode'], df['Return_moving_avg'], label=f'PPO')


plt.xlabel('Episode', fontsize=15)
plt.ylabel('Return', fontsize=15)
plt.title('Moving average of return for the different algorithms', fontsize=15)
plt.legend(fontsize=13)
plt.grid(True)
plt.show()