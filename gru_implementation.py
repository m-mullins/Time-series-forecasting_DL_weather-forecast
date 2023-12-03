import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import time

from utilities.utilities import split_sequences
from GRU.gru_model import GRU

# Record the start time
start_time = time.time()

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
chosen_model = "GRU"               # Choose which model structure to use
epochs = 150                        # Training epochs
dropout = .0                        # Dropout
learning_rate = 0.005               # Learning rate
num_classes =   TS_FUTURE
input_size = num_features
hidden_size = 32
num_layers = 1
output_dim = TS_FUTURE

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

# Initialize model
model_params = {
        'num_classes':  num_classes,
        'input_size':   input_size,
        'hidden_size':  hidden_size,
        'num_layers':   num_layers,
        'dropout':      dropout,
    }
model = GRU(**model_params)
print(model)

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
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Adjust max_norm as needed
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
        
# Record the end time
end_time = time.time()

# Calculate the elapsed time
train_time = round(end_time - start_time,1)
print(f"Execution time: {train_time} seconds")

# Training progress
plt.title('Training Progress')
plt.yscale("log")
plt.plot(training_loss, label = 'train')
plt.plot(validation_loss, label = 'validation')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plot_results_directory = os.path.dirname(os.path.abspath(__file__)) + "\\GRU\\Results"
figure_path = os.path.join(plot_results_directory, "Training_progress.png")
plt.savefig(figure_path)
plt.show()

# Load best trained model
best_model = GRU(**model_params)
best_model.load_state_dict(best_params)
torch.save(best_model,'GRU\\gru_trained_model_' + str(FEATURE_FORECASTED) + '.pt')
best_model.eval()


# Calculate test loss
gru_prediction = best_model(x_test)
gru_mse_loss = round(mse_loss(gru_prediction, y_test).item(), 4)
print(f"GRU mse loss: {gru_mse_loss}")

# Test model
gru_prediction = best_model(x_test[-1].unsqueeze(0))
gru_prediction = gru_prediction.detach().numpy()
gru_prediction = gru_prediction * z_std[selected_feature] + z_mean[selected_feature]    # Remove z-score normalization
gru_prediction = gru_prediction[0].tolist()

# Test target
test_target = y_arr_test[-1,:]
test_target = test_target * z_std[selected_feature] + z_mean[selected_feature]  # Remove z-score normalization
test_target = test_target.tolist()

# Plot prediction
plt.plot(test_target, label="Actual Data")
plt.plot(gru_prediction, label="GRU Predictions")
plt.title('GRU predictions')
plt.ylabel(f"{feature_list[selected_feature%5]}")
plt.xlabel("Time")
plt.legend()
figure_path = os.path.join(plot_results_directory, "GRU_predictions.png")
plt.savefig(figure_path)
plt.show()

# Save parameters and results to csv
csv_file_path = os.path.join(plot_results_directory, "parameter_iterations.csv")
# Check if the CSV file exists
if not os.path.isfile(csv_file_path):
    # If the file does not exist, create a new DataFrame
    df_params_iter = pd.DataFrame(columns=['STATION_FORECASTED', 'selected_feature', 'model_type', 'epochs', 'learning_rate', 'num_classes', 'input_size', 'hidden_size','num_layers', 'train_time', 'gru_mse_loss'])
else:
    # If the file exists, load the existing DataFrame
    df_params_iter = pd.read_csv(csv_file_path)
new_row = {'STATION_FORECASTED': STATION_FORECASTED,
           'selected_feature': selected_feature,
           'model_type' : chosen_model,
           'epochs': epochs,
           'learning_rate': learning_rate,
           'num_classes':   num_classes,
           'input_size':  input_size,
           'hidden_size': hidden_size,
           'num_layers':  num_layers,
           'train_time' : train_time,
           'gru_mse_loss': gru_mse_loss}
df_params_iter = pd.concat([df_params_iter, pd.DataFrame([new_row])], ignore_index=True)
df_params_iter.to_csv(csv_file_path, index=False)

# Plot prediction and context
plt.figure(figsize=(10,6)) #plotting
plt.title('GRU predictions with context')
start_plot = len(y_arr_test) - 2*TS_PAST
y = df.iloc[:,selected_feature]
y = y[-len(y_arr_test):]
y = y * z_std[selected_feature] + z_mean[selected_feature]  # Remove z-score normalization
a = [x for x in range(start_plot, len(y))]
plt.plot(a, y[start_plot:], label='Actual data')
c = [x for x in range(len(y)-TS_FUTURE, len(y))]
plt.plot(c, gru_prediction, label=f'One-shot multi-step prediction ({TS_FUTURE}h)')
plt.axvline(x=len(y)-TS_FUTURE, c='r', linestyle='--')
plt.ylabel(f"{feature_list[selected_feature%5]}")
plt.xlabel("Time")
plt.legend(loc='upper left')
figure_path = os.path.join(plot_results_directory, "GRU_predictions_context.png")
text_content = '\n'.join([f'{key}: {value}' for key, value in new_row.items()])
plt.text(0.05, 0.05, text_content, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom', horizontalalignment='left')
plt.savefig(figure_path)
plt.show()

print("Done")
