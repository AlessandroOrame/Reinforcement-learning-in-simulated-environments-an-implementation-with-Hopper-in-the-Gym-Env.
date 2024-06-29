import pandas as pd
import matplotlib.pyplot as plt
import glob

# Trova tutti i file CSV nella directory corrente
file_paths = glob.glob('log/*.csv')

# Lista per salvare i DataFrame
dataframes = []

# Carica ogni file CSV in un DataFrame, specificando i nomi delle colonne
for file_path in file_paths:
    df = pd.read_csv(file_path, names=['Episode', 'Return'], skiprows=1)
    dataframes.append(df)

# Plotta i dati
plt.figure(figsize=(10, 6))

for i, df in enumerate(dataframes):
    #df['Return_moving_avg'] = df['Return'].rolling(window=5).mean()
    if i == 0:
        plt.plot(df['Episode'], df['Return'], label='Reinforce')
    elif i == 1:
        plt.plot(df['Episode'], df['Return'], label=f'Actor-Critic')
    else:
        plt.plot(df['Episode'], df['Return'], label=f'Reinforce with baseline=20')


plt.xlabel('Episode', fontsize=15)
plt.ylabel('Return', fontsize=15)
plt.title('Average behavior of algorithms during training', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
