import torch
import torch.nn as nn

class SimpleMNISTModel(nn.Module):
    def __init__(self, dropout_rate=0):
        super(SimpleMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256)  ,
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10) ,
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
