import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x