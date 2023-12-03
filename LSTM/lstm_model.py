import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = True

        # Unidirectional LSTM model with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state for each layer and batch
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through Unidirectional LSTM with dropout
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Reshape hidden states for Dense layer
        hn = hn.view(self.num_layers, -1, self.hidden_size)
        out = self.relu(hn[-1])  # take the output from the last layer
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out
    

class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        self.batch_first = True

        # Bi-LSTM model with one dropout layer after BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            bidirectional=True
        )
        self.dropout_lstm = nn.Dropout(p=dropout)
        self.fc_1 = nn.Linear(hidden_size * 2, 128)  # fully connected, multiply by 2 for bidirectional
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state for each layer and batch
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # multiply by 2 for bidirectional
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # multiply by 2 for bidirectional

        # Propagate input through Bi-LSTM with one dropout layer
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        output = self.dropout_lstm(output)
        hn = hn.view(self.num_layers, -1, self.hidden_size * 2)  # reshape hidden states for Dense layer, multiply by 2 for bidirectional
        out = self.relu(hn[-1])  # take the output from the last layer
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out
    
# Bi-LSTM alternatives tried:
# Add dropout after each layer, lower learning rate, add gradient clipping, implement mini-batch learning