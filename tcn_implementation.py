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
STATION_FORCASTED = 0
feature_list = ['Temp (degC)','Rel Hum (%)','Precip. Amount (mm)','Stn Press (kPa)','Wind Spd (km/h)']
FEATURE_FORECASTED = 0
selected_feature = len(feature_list)*STATION_FORCASTED+FEATURE_FORECASTED
num_features = len(stations) * len(feature_list)
# Time steps to look into the past (context)
TS_PAST = 120   # [h]
# Time steps to look into the future (forecast)
TS_FUTURE = 24   # [h]

# Global nn parameters
# training epochs
epochs = 150
# test dataset size
test_len = 300
# Temporal causal layer channels
channel_sizes = [10] * num_features
# Convolution kernel size
kernel_size = 5
dropout = .0

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

# Preprocessing with z-score normalization
df = (df - df.mean()) / df.std()

# Feed context and prediction results to the model (adding another dimension)
x_arr,y_arr = split_sequences(df,df.iloc[:,selected_feature],TS_PAST, TS_FUTURE)

# Datasplit properties
frac_train = 0.7
frac_val = 0.15
num_samples = x_arr.shape[0]
num_train = int(frac_train * num_samples)
num_val = int(frac_val * num_samples)
num_test = num_samples - num_train - num_val

# Split the data into train/validate/test
x_arr_train = x_arr[:num_train, :, :]
x_arr_val = x_arr[num_train:num_train + num_val, :, :]
x_arr_test = x_arr[num_train + num_val:, :, :]
print(f"\nx_arr shapes:\n{x_arr_train.shape}\n{x_arr_val.shape}\n{x_arr_test.shape}")
y_arr_train = y_arr[:num_train, :]
y_arr_val = y_arr[num_train:num_train + num_val, :]
y_arr_test = y_arr[num_train + num_val:, :]
print(f"\ny_arr shapes:\n{y_arr_train.shape}\n{y_arr_val.shape}\n{y_arr_test.shape}")

# We want to feed 120 samples and predict the next 24 !!!!!!!!!!!!!!!!
# Tensor shape sould be x_train(12264,120,15) and y_train(12264,24)
# x[TS length,TS context, num features] and y[TS length,TS forecast]
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

x_train = x_train.permute(0,2,1)
x_val = x_val.permute(0,2,1)
x_test = x_test.permute(0,2,1)

# Initialize model
model_params = {
    'input_size':   num_features, # Number of features
    'output_size':  1,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout
}
model = TCN(**model_params)
# print(model)
# model = LSTM(TS_FUTURE,num_features,2,1)

# Define optimizer and loss functions
optimizer = torch.optim.Adam(params = model.parameters(), lr = .005)
mse_loss = torch.nn.MSELoss()

# Training
best_params = None
min_val_loss = sys.maxsize

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
plt.show()

best_model = TCN(**model_params)
best_model.eval()
best_model.load_state_dict(best_params)

tcn_prediction = best_model(x_test)
dummy_prediction = Dummy()(x_test)

tcn_mse_loss = round(mse_loss(tcn_prediction, y_test).item(), 4)
dummy_mse_loss = round(mse_loss(dummy_prediction, y_test).item(), 4)

# plt.title(f'Test| TCN: {tcn_mse_loss}; Dummy: {dummy_mse_loss}')
# plt.plot(
#     ts_int(
#         tcn_prediction.view(-1).tolist(),
#         ts[-test_len:, 0],
#         start = ts[-test_len - 1, 0]
#     ),
#     label = 'tcn')
# plt.plot(ts[-test_len - 1:, 0], label = 'real')
# plt.legend()
# plt.show()
# plt.savefig("Model MSE loss.png")

print("Done")