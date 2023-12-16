# Time Series Forecasting with Deep Learning for Weather Forecasting

## Introduction
This repository contains code for a time series forecasting model that compares deep learning models (LSTM, GRU, TCN) to predict weather conditions. The goal is to provide accurate forecasts based on historical weather data from the Government of Canada's historical weather data.

## Features
### Data Preparation: 
Folder : get_weather_data

download_weather_data.py : Download historical weather data from https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/ and store in csv files.

data_cleaner.py : Clean raw data to keep selected features and concatenate to pkl and csv files.

### Model architecture:
Folders : LSTM, GRU, TCN

Files (model)_model.py contain the classes creating the model architectures.

### Training:
Files (model)_implementation.py define the hyperparameters, train the model, save the trained weights and show initial results.

### Results:
File test_trained_models.py allows to test and compare all trained models.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
