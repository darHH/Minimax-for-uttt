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

# Define three loss functions
criterion_mse   = nn.MSELoss()
criterion_huber = nn.HuberLoss()  # Huber Loss (also known as SmoothL1Loss)
criterion_mae   = nn.L1Loss()      # Mean Absolute Error Loss

# Choose which loss function to use: options are 'mse', 'huber', or 'mae'
loss_type = 'huber'  # Change this to 'mse' or 'mae' as needed

if loss_type == 'mse':
    criterion = criterion_mse
elif loss_type == 'huber':
    criterion = criterion_huber
elif loss_type == 'mae':
    criterion = criterion_mae
else:
    raise ValueError("Invalid loss type. Choose 'mse', 'huber', or 'mae'.")

optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

# Training loop
epochs, batch_size = 80, 256
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / (X_train.size(0) // batch_size)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.5f}")

# Validate model
model.eval()
with torch.no_grad():
    val_loss = criterion(model(X_val), y_val).item()
print(f"Validation Loss ({loss_type}): {val_loss:.5f}")

# Save trained model weights
torch.save(model.state_dict(), "trained_uttt_model.pth")
print("Finished Training")