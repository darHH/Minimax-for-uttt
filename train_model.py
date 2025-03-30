import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from data_preparation import prepare_data
from uttt_model import UTTTEvaluator
import numpy as np
from utils import State, Action, load_data


# Load your original data here:
validation_data = load_data()
states_array = np.array([board for board, val in validation_data])  # shape: (80000,3,3,3,3)
evaluations_array = np.array([val for board, val in validation_data])  # shape: (80000,)
X, y = prepare_data(states_array, evaluations_array)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model setup
model = UTTTEvaluator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00019, weight_decay=1e-4)

# Training loop
epochs, batch_size = 100, 256
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])

    epoch_loss = 0
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/(X_train.size(0)//batch_size):.5f}")

# Validate model
model.eval()
with torch.no_grad():
    val_loss = criterion(model(X_val), y_val).item()
print(f"Validation MSE: {val_loss:.5f}")

# Save trained model weights
torch.save(model.state_dict(), "trained_uttt_model.pth")
print("Finished Training")