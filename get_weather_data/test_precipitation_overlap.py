# This script plots the precipitation amount over time of each cleaned pickle file

import os
import pandas as pd
import matplotlib.pyplot as plt

# Concat all pickles
df_list = []
files = os.listdir(os.path.join(".", "pickles"))
for file in files:
    target_file = ".\\pickles\\" + file
    df = pd.read_pickle(target_file)
    df_list.append(df)

# print(df_list)

# Plot
for index in range(3):
    plt.plot(df_list[index]['Date/Time (LST)'][:1000], df_list[index]['Precip. Amount (mm)'][:1000], label=f'DataFrame {index+1}')
plt.xlabel('Date/Time (LST)')
plt.ylabel('Precip. Amount (mm)')
plt.title('Precipitation Over Time')
plt.legend()
plt.show()