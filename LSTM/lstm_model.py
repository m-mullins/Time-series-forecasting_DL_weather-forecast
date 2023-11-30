import torch.nn as nn
import torch

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.0, bidirectional=False) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out
    

class Bi_LSTM_old(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, output_dim, num_layers=2, rnn_type='LSTM'):
        super(Bi_LSTM_old, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_size, self.input_size)

        # Define the LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_size * 2, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, input):
        #Forward pass through initial hidden layer
        linear_input = self.init_linear(input)

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size ,hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        lstm_out, self.hidden = self.lstm(linear_input)

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        return y_pred
    

class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        self.batch_first = True
        # Bi-LSTM model with dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=self.batch_first, dropout=dropout_rate, bidirectional=True)  # bidirectional
        self.dropout_lstm = nn.Dropout(p=dropout_rate)
        self.fc_1 = nn.Linear(hidden_size * 2, 128)  # fully connected, multiply by 2 for bidirectional
        self.dropout_fc_1 = nn.Dropout(p=dropout_rate)
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.dropout_fc_2 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # multiply by 2 for bidirectional
        # cell state
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # multiply by 2 for bidirectional
        # propagate input through Bi-LSTM with dropout
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        output = self.dropout_lstm(output)
        hn = hn.view(-1, self.hidden_size * 2)  # reshaping for Dense layer next, multiply by 2 for bidirectional
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.dropout_fc_1(out)
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        out = self.dropout_fc_2(out)
        return out
    
# Bi-LSTM alternatives tried:
# Add dropout after each layer, lower learning rate, add gradient clipping, implement mini-batch learning