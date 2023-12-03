import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import csv

from utilities.utilities import split_sequences, update_results_in_csv
from LSTM.lstm_model import LSTM
from TCN.tcn_model import TCN
from sklearn.metrics import mean_absolute_error, r2_score


# Station and feature that we want to forecast
stations = [30165,48374,49608]
STATION_FORECASTED = 0
feature_list = ['Temp (degC)','Rel Hum (%)','Precip. Amount (mm)','Stn Press (kPa)','Wind Spd (km/h)']
FEATURE_FORECASTED = 4
selected_feature = len(feature_list)*STATION_FORECASTED+FEATURE_FORECASTED
num_features = len(stations) * len(feature_list)

# Choose model
model_list = ['TCN','LSTM','GRU']
chosen_model = model_list[0]

# Context and forecast length# 
TS_PAST     = 120   # Time steps to look into the past (context) [h]
TS_FUTURE   = 24    # Time steps to look into the future (forecast) [h]

# Import time-series from stored pickles
df_list = []
files = os.listdir(os.path.join(".", "data\\pickles"))
for file in files:
    target_file = ".\\data\\pickles\\" + file
    df = pd.read_pickle(target_file)
    df = df.drop(columns='Date/Time (LST)')
    df_list.append(df)
nb_stations = len(files)
df = pd.concat(df_list,axis=1)
df = df.iloc[40000:60000,:] # crop df to lower calc time

# Preprocessing data with z-score normalization
z_mean = df.mean()
z_std = df.std()
df = (df - z_mean) / z_std

# Feed context and prediction results to the model (adding another dimension)
x_arr,y_arr = split_sequences(df,df.iloc[:,selected_feature],TS_PAST, TS_FUTURE)

# Datasplit properties
frac_train  = 0.7
frac_val    = 0.15
num_samples = x_arr.shape[0]
num_train   = int(frac_train * num_samples)
num_val     = int(frac_val * num_samples)
num_test    = num_samples - num_train - num_val

# Split the data into train/validate/test
x_arr_train = x_arr[:num_train, :, :]
x_arr_val   = x_arr[num_train:num_train + num_val, :, :]
x_arr_test  = x_arr[num_train + num_val:, :, :]
print(f"\nx_arr shapes:\n{x_arr_train.shape}\n{x_arr_val.shape}\n{x_arr_test.shape}")
y_arr_train = y_arr[:num_train, :]
y_arr_val   = y_arr[num_train:num_train + num_val, :]
y_arr_test  = y_arr[num_train + num_val:, :]
print(f"\ny_arr shapes:\n{y_arr_train.shape}\n{y_arr_val.shape}\n{y_arr_test.shape}")

# We want to feed 120 samples and predict the next 24 for 15 different features
# Tensor shape sould be x_train(12264,120,15) and y_train(12264,24)
# x[TS batch size,TS context, num features] and y[TS batch size,TS forecast]
x_train = torch.tensor(x_arr_train, dtype=torch.float32)
x_val   = torch.tensor(x_arr_val, dtype=torch.float32)
x_test  = torch.tensor(x_arr_test, dtype=torch.float32)
y_train = torch.tensor(y_arr_train, dtype=torch.float32)
y_val   = torch.tensor(y_arr_val, dtype=torch.float32)
y_test  = torch.tensor(y_arr_test, dtype=torch.float32)
train_len = x_train.size()[0]
print(f"\nTensor shapes (x):\n{x_train.shape}\n{x_val.shape}\n{x_test.shape}")
print(f"\nTensor shapes (y):\n{y_train.shape}\n{y_val.shape}\n{y_test.shape}\n")

# Load model
if chosen_model == 'TCN':
    # Global nn parameters
    epochs = 150                        # Training epochs
    input_size = TS_PAST                # Context
    output_size = TS_FUTURE             # Forecast
    channel_sizes = [num_features]*5    # Temporal causal layer channels [num of features]*amount of filters per layer
    kernel_size = 5                     # Convolution kernel size
    dropout = .3                        # Dropout
    learning_rate = 0.005               # Learning rate

    model_params = {
        'input_size':   input_size,
        'output_size':  output_size,
        'num_channels': channel_sizes,
        'kernel_size':  kernel_size,
        'dropout':      dropout
    }
    # Load the saved state_dict into the model
    best_model = TCN(**model_params)
    best_model.load_state_dict(torch.load('TCN\\tcn_trained_model_' + str(FEATURE_FORECASTED) + '.pt'))
    best_model.eval()

else:
    model_path = chosen_model + '\\' + str.lower(chosen_model) + '_trained_model_' + str(selected_feature) + '.pt'
    best_model = torch.load(model_path)
    best_model.eval()

# Calculate MSE, MAE and R^2 score
mse_loss            = torch.nn.MSELoss()
model_prediction    = best_model(x_test)
model_mse_loss      = round(mse_loss(model_prediction, y_test).item(), 4)
mae_loss            = mean_absolute_error(y_test.cpu().detach().numpy(), model_prediction.cpu().detach().numpy())
model_mae_loss      = round(mae_loss, 4)
r2_loss             = r2_score(y_test.cpu().detach().numpy(), model_prediction.cpu().detach().numpy())
model_r2_loss       = round(r2_loss, 4)
print(f"{chosen_model} MSE loss: {model_mse_loss}")
print(f"{chosen_model} MAE loss: {model_mae_loss}")
print(f"{chosen_model} R^2 score: {model_r2_loss}")

# Save losses to csv
csv_file_path = 'loss_results.csv'
update_results_in_csv(csv_file_path, chosen_model, feature_list, FEATURE_FORECASTED, model_mse_loss, model_mae_loss, model_r2_loss)

# Test model
model_prediction = best_model(x_test[-1].unsqueeze(0))
model_prediction = model_prediction.detach().numpy()
model_prediction = model_prediction * z_std[selected_feature] + z_mean[selected_feature]    # Remove z-score normalization
model_prediction = model_prediction[0].tolist()

# Test target
test_target = y_arr_test[-1,:]
test_target = test_target * z_std[selected_feature] + z_mean[selected_feature]  # Remove z-score normalization
test_target = test_target.tolist()

# Plot prediction
plot_results_directory = os.path.dirname(os.path.abspath(__file__)) + "\\" + chosen_model + "\\Results"
plt.plot(test_target, label="Actual Data")
plt.plot(model_prediction, label=f"{chosen_model} Predictions")
plt.title(f'{chosen_model} predictions')
plt.ylabel(f"{feature_list[selected_feature%5]}")
plt.xlabel("Time")
plt.legend()
figure_path = os.path.join(plot_results_directory, chosen_model + "_predictions.png")
plt.savefig(figure_path)
plt.show()

# Plot prediction and context
plt.figure(figsize=(10,6)) #plotting
plt.title(f'{chosen_model} predictions with context')
start_plot = len(y_arr_test) - 2*TS_PAST
y = df.iloc[:,selected_feature]
y = y[-len(y_arr_test):]
y = y * z_std[selected_feature] + z_mean[selected_feature]  # Remove z-score normalization
a = [x for x in range(start_plot, len(y))]
plt.plot(a, y[start_plot:], label='Actual data')
c = [x for x in range(len(y)-TS_FUTURE, len(y))]
plt.plot(c, model_prediction, label=f'One-shot multi-step prediction ({TS_FUTURE}h)')
plt.axvline(x=len(y)-TS_FUTURE, c='r', linestyle='--')
plt.ylabel(f"{feature_list[selected_feature%5]}")
plt.xlabel("Time")
plt.legend(loc='upper left')
# Add parameters as text box in plot
new_row = {'STATION_FORECASTED': STATION_FORECASTED,
           'selected_feature': selected_feature,
           'model_type' : chosen_model,
           'mse_loss': model_mse_loss}
figure_path = os.path.join(plot_results_directory, chosen_model + "_predictions_context.png")
text_content = '\n'.join([f'{key}: {value}' for key, value in new_row.items()])
plt.text(0.05, 0.05, text_content, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom', horizontalalignment='left')
plt.savefig(figure_path)
plt.show()

print("Done")
