import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
params = params_df.iloc[param_row]

num_layers = params['num_layers']
layer_size = params['layer_size']
activation_str = params['activation']

# Print the stat
print(f'{num_layers} layer -- {layer_size} neurons -- {activation_str} activation')

# Define activation function
if activation_str == 'relu':
    activation = nn.ReLU()
else:
    activation = nn.Tanh()

# Load train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Split train data into train and validation sets (70% train, 30% validation)
train, val = train_test_split(train_data, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
x_train = torch.tensor(train[['x1', 'x2']].values, dtype=torch.float32)
y_train = torch.tensor(train['y'].values, dtype=torch.float32).unsqueeze(1)
x_val = torch.tensor(val[['x1', 'x2']].values, dtype=torch.float32)
y_val = torch.tensor(val['y'].values, dtype=torch.float32).unsqueeze(1)
x_test = torch.tensor(test_data[['x1', 'x2']].values, dtype=torch.float32)
y_test = torch.tensor(test_data['y'].values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)

# Define MLP model class
class MLP(nn.Module):
    def __init__(self, input_size, layer_size, num_layers, activation):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(activation)
        layers.append(nn.Linear(layer_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Initialize model
model = MLP(input_size=2, layer_size=layer_size, num_layers=num_layers, activation=activation)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
best_val_loss = float('inf')
best_model_state = None  # To store the best model in memory
patience = 100
epochs_no_improve = 0
max_epochs = 100000
stop_epoch = None  # To record the epoch at which training stops

# Training loop
for epoch in range(max_epochs):
    model.train()
    running_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Check early stopping and store best model in memory
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save the best model state in memory
        epochs_no_improve = 0
        stop_epoch = epoch  # Update the stopping epoch
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        break

# Test set evaluation with the best model (from memory)
model.load_state_dict(best_model_state)  # Load the best model from memory
model.eval()
test_loss = 0
with torch.no_grad():
    test_outputs = model(x_test)
    test_loss = criterion(test_outputs, y_test).item()

# Save the results
results = {
    'num_layers': num_layers,
    'layer_size': layer_size,
    'activation': activation_str,
    'train_loss': running_loss / len(train_loader),
    'val_loss': best_val_loss,
    'test_loss': test_loss,
    'stop_epoch': stop_epoch  # Record the epoch at which training stopped
}

results_df = pd.DataFrame([results])
results_df.to_csv(f'results/{param_row}.csv', index=False)

print(f"Done {param_row}")
