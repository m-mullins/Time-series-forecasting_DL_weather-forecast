import pandas as pd

# Assuming df is your time-series DataFrame
# Replace this with the actual initialization of your DataFrame
# For example, df = pd.read_csv('your_file.csv')

# Sample DataFrame for demonstration
n = 13
data = {'Timestamp': pd.date_range(start='2022-01-01', periods=n), 'Value': range(n)}
df = pd.DataFrame(data)

# Proportions for training, validation, and test sets
frac_train = 0.7
frac_val = 0.15

# Calculate cumulative sum of proportions
cumulative_sum = (frac_train, frac_train + frac_val, 1.0)

# Ensure the cumulative sum does not exceed 1.0
assert cumulative_sum[-1] == 1.0, "Proportions do not sum to 1.0"

# Calculate indices to split the data
split_indices = (df.shape[0] * pd.Series(cumulative_sum)).astype(int)

# Split the data
df_train = df.loc[:split_indices[0] - 1]
df_val = df.loc[split_indices[0]:split_indices[1] - 1]
df_test = df.loc[split_indices[1]:]

# Display the lengths of each set
print("Training set length:", len(df_train))
print("Validation set length:", len(df_val))
print("Test set length:", len(df_test))
