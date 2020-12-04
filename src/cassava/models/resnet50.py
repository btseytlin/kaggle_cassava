import torch
from torch import nn


class ResnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        trunk = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        head = nn.Linear(trunk.fc.in_features, 5)

        self.trunk = trunk
        self.trunk.fc = head
        self.head = self.trunk.fc

    def forward(self, x):
        return self.trunk.forward(x)

    def predict(self, x):
        logits = self.forward(x)
        probabilities = nn.functional.softmax(logits, dim=0)
        return probabilities

    def predict_label(self, x):
        return torch.max(self.predict(x), 1)[1]
