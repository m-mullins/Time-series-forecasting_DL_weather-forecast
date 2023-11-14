import numpy as np
import pandas as pd

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
