import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.2):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = True

        # Unidirectional GRU model with dropout
        self.gru = nn.GRU(
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

        # Propagate input through Unidirectional GRU with dropout
        output, h_n = self.gru(x, h_0)

        # Reshape hidden states for Dense layer
        h_n = h_n.view(self.num_layers, -1, self.hidden_size)
        out = self.relu(h_n[-1])  # take the output from the last layer
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out
