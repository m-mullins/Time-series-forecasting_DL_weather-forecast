import numpy as np
import pandas as pd
import csv

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    # Splits a multivariate input sequence into past and future samples
    # input_sequences   : multivariate sequence inputed (x)
    # output_sequence   : multivariate sequence outputed (y)
    # n_steps_in        : number of time steps to look back
    # n_steps_out       : number of time steps to look in the future

    X, y = list(), list() # instantiate X and y

    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)

    return np.array(X), np.array(y)


def update_results_in_csv(csv_file_path, chosen_model, feature_list, FEATURE_FORECASTED, model_mse_loss, model_mae_loss, model_r2_loss):
    # Updates the MSE, MAE and R2 losses in the loss_results csv file for the given model and feature

    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Replace spaces with underscores to create valid column names
    chosen_model_no_space = chosen_model.replace(' ', '_')

    # Find the index of the chosen feature in the feature list
    chosen_feature_index = FEATURE_FORECASTED

    # Update the values for the chosen model and feature
    df.at[chosen_feature_index, f"{chosen_model_no_space}_MSE"] = round(model_mse_loss, 4)
    df.at[chosen_feature_index, f"{chosen_model_no_space}_MAE"] = round(model_mae_loss, 4)
    df.at[chosen_feature_index, f"{chosen_model_no_space}_R2"] = round(model_r2_loss, 4)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Results for {chosen_model} and {feature_list[chosen_feature_index]} updated in {csv_file_path}")
