import torch
import torch.nn as nn
import torch.nn.functional as F

class UTTTEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(243, 128)
        self.dropout1 = nn.Dropout(p=0.5)  # 50% dropout; adjust as needed
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.tanh(self.fc3(x))  # Predict in range [-1,1]