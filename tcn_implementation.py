import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

from TCN.dummy import Dummy
from TCN.tcn_model import TCN
from utilities.utilities import split_sequences
from LSTM.lstm_model import LSTM


# Station and feature that we want to forecast
stations = [30165,48374,49608]
STATION_FORECASTED = 0
feature_list = ['Temp (degC)','Rel Hum (%)','Precip. Amount (mm)','Stn Press (kPa)','Wind Spd (km/h)']
FEATURE_FORECASTED = 0
selected_feature = len(feature_list)*STATION_FORECASTED+FEATURE_FORECASTED
num_features = len(stations) * len(feature_list)

# Context and forecast length# 
TS_PAST     = 120   # Time steps to look into the past (context) [h]
TS_FUTURE   = 24    # Time steps to look into the future (forecast) [h]

# Global nn parameters
epochs = 300                        # Training epochs
input_size = TS_PAST                # Context
output_size = TS_FUTURE             # Forecast
channel_sizes = [num_features]*5    # Temporal causal layer channels
kernel_size = 5                     # Convolution kernel size
dropout = .3                        # Dropout
learning_rate = 0.005               # Learning rate

# Import time-series from stored pickles
df_list = []
files = os.listdir(os.path.join(".", "pickles"))
for file in files:
    target_file = ".\\pickles\\" + file
    df = pd.read_pickle(target_file)
    df = df.drop(columns='Date/Time (LST)')
    df_list.append(df)
nb_stations = len(files)
df = pd.concat(df_list,axis=1)

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
# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130
x_train = torch.tensor(x_arr_train, dtype=torch.float32)
x_val   = torch.tensor(x_arr_val, dtype=torch.float32)
x_test  = torch.tensor(x_arr_test, dtype=torch.float32)
y_train = torch.tensor(y_arr_train, dtype=torch.float32)
y_val   = torch.tensor(y_arr_val, dtype=torch.float32)
y_test  = torch.tensor(y_arr_test, dtype=torch.float32)
train_len = x_train.size()[0]
print(f"\nTensor shapes (x):\n{x_train.shape}\n{x_val.shape}\n{x_test.shape}")
print(f"\nTensor shapes (y):\n{y_train.shape}\n{y_val.shape}\n{y_test.shape}")

# Initialize model
model_params = {
    'input_size':   input_size,
    'output_size':  output_size,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout
}
model = TCN(**model_params)
# print(model)
# model = LSTM(TS_FUTURE,num_features,2,1)

# Define optimizer and loss functions
optimizer   = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
mse_loss    = torch.nn.MSELoss()

# Training
best_params     = None
min_val_loss    = sys.maxsize

training_loss = []
validation_loss = []

for epoch in range(epochs):

    # Calculate training loss
    prediction = model(x_train)
    loss = mse_loss(prediction, y_train)

    optimizer.zero_grad()   # Calculate gradient, manually setting to 0
    loss.backward()         # Calculate loss from loss function
    optimizer.step()        # Improve from loss (backprop)

    # Calculate validation loss
    val_prediction = model(x_val)
    val_loss = mse_loss(val_prediction, y_val)

    # Append losses to loss list
    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    
    # Copy current epoch param if they're the best yet
    if val_loss.item() < min_val_loss:
        best_params = copy.deepcopy(model.state_dict())
        min_val_loss = val_loss.item()

    # Print results for every 100th epoc
    if epoch % 10 == 0:
        diff = (y_train - prediction).view(-1).abs_().tolist()
        print(f'epoch {epoch}. train: {round(loss.item(), 4)}, '
              f'val: {round(val_loss.item(), 4)}')
        
# Training progress
plt.title('Training Progress')
plt.yscale("log")
plt.plot(training_loss, label = 'train')
plt.plot(validation_loss, label = 'validation')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plot_results_directory = os.path.dirname(os.path.abspath(__file__)) + "\\TCN\\Results"
figure_path = os.path.join(plot_results_directory, "Training_progress.png")
plt.savefig(figure_path)
plt.show()

# Load best trained model
best_model = TCN(**model_params)
# best_model = LSTM(TS_FUTURE,num_features,2,1)
best_model.eval()
best_model.load_state_dict(best_params)

# Calculate test loss
tcn_prediction = best_model(x_test)
tcn_mse_loss = round(mse_loss(tcn_prediction, y_test).item(), 4)
print(f"TCN mse loss: {tcn_mse_loss}")

# Test model
tcn_prediction = best_model(x_test[-1].unsqueeze(0))
tcn_prediction = tcn_prediction.detach().numpy()
tcn_prediction = tcn_prediction * z_std[selected_feature] + z_mean[selected_feature]    # Remove z-score normalization
tcn_prediction = tcn_prediction[0].tolist()
# dummy_prediction = Dummy()(x_test)
# dummy_mse_loss = round(mse_loss(dummy_prediction, y_test).item(), 4)

# Test target
test_target = y_arr_test[-1,:]
test_target = test_target * z_std[selected_feature] + z_mean[selected_feature]  # Remove z-score normalization
test_target = test_target.tolist()

# Plot prediction
plt.plot(test_target, label="Actual Data")
plt.plot(tcn_prediction, label="TCN Predictions")
plt.title('TCN predictions')
plt.ylabel(f"{feature_list[selected_feature%5]}")
plt.xlabel("Time")
plt.legend()
figure_path = os.path.join(plot_results_directory, "TCN_predictions.png")
plt.savefig(figure_path)
plt.show()

# Save parameters and results to csv
csv_file_path = os.path.join(plot_results_directory, "parameter_iterations.csv")
# Check if the CSV file exists
if not os.path.isfile(csv_file_path):
    # If the file does not exist, create a new DataFrame
    df_params_iter = pd.DataFrame(columns=['STATION_FORECASTED', 'selected_feature', 'epochs', 'learning_rate', 'channel_sizes', 'kernel_size', 'dropout', 'tcn_mse_loss'])
else:
    # If the file exists, load the existing DataFrame
    df_params_iter = pd.read_csv(csv_file_path)
new_row = {'STATION_FORECASTED': STATION_FORECASTED,
           'selected_feature': selected_feature,
           'epochs': epochs,
           'learning_rate': learning_rate,
           'channel_sizes': channel_sizes,
           'kernel_size': kernel_size,
           'dropout': dropout,
           'tcn_mse_loss': tcn_mse_loss}
df_params_iter = pd.concat([df_params_iter, pd.DataFrame([new_row])], ignore_index=True)
df_params_iter.to_csv(csv_file_path, index=False)

# Plot prediction and context
plt.figure(figsize=(10,6)) #plotting
plt.title('TCN predictions with context')
start_plot = 2500
y = df.iloc[:,selected_feature]
y = y[-len(y_arr_test):]
y = y * z_std[selected_feature] + z_mean[selected_feature]  # Remove z-score normalization
a = [x for x in range(start_plot, len(y))]
plt.plot(a, y[start_plot:], label='Actual data')
c = [x for x in range(len(y)-TS_FUTURE, len(y))]
plt.plot(c, tcn_prediction, label=f'One-shot multi-step prediction ({TS_FUTURE}h)')
plt.axvline(x=len(y)-TS_FUTURE, c='r', linestyle='--')
plt.ylabel(f"{feature_list[selected_feature%5]}")
plt.xlabel("Time")
plt.legend()
figure_path = os.path.join(plot_results_directory, "TCN_predictions_context.png")
text_content = '\n'.join([f'{key}: {value}' for key, value in new_row.items()])
plt.text(0.05, 0.05, text_content, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom', horizontalalignment='left')
plt.savefig(figure_path)
plt.show()

print("Done")
