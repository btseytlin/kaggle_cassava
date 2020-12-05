import torch
from torch import nn
import torchvision.models as models


class ResnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        trunk = models.resnet18(pretrained=True)
        head = nn.Linear(trunk.fc.in_features, 5)

        self.trunk = trunk
        self.trunk.fc = head
        self.head = self.trunk.fc

    def forward(self, x):
        return self.trunk.forward(x)

    def predict(self, x):
        logits = self.forward(x)
        probabilities = nn.functional.softmax(logits, dim=1)
        return probabilities

    def predict_label(self, x):
        return torch.max(self.predict(x), 1)[1]
