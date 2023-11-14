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
feature_list = ['Date/Time (LST)','Temp (degC)','Rel Hum (%)','Precip. Amount (mm)','Stn Press (kPa)','Wind Spd (km/h)']
feature = feature_list[1]

# Plot
for index in range(3):
    plt.plot(df_list[index][feature_list[0]][:1000], df_list[index][feature][:1000], label=f'DataFrame {index+1}')
plt.xlabel(feature_list[0])
plt.ylabel(feature)
plt.title(feature + ' Over Time')
plt.legend()
plt.show()