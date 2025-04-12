import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic supermarket sales data
np.random.seed(42)
torch.manual_seed(42)

num_samples = 1000
num_features = 4  # Example features: Advertising, Customer Footfall, Product Price, Discounts

# Generate random values for the features
X = np.random.rand(num_samples, num_features) * 100  # Scale values between 0 and 100

# Generate sales target using a linear relationship with some noise
y = 5 * X[:, 0] + 3 * X[:, 1] - 2 * X[:, 2] + 4 * X[:, 3] + np.random.randn(num_samples) * 10

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = (y - y.mean()) / y.std()  # Normalize target

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test,
                                                                                       dtype=torch.float32).view(-1, 1)


# Define the neural network model for linear regression
class SalesPredictionNN(nn.Module):
    def __init__(self, input_dim):
        super(SalesPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)  # Output layer (regression task)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (linear output)
        return x


# Initialize model, loss function, and optimizer
model = SalesPredictionNN(num_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Display weights before training
print("Weights before training:")
for name, param in model.named_parameters():
    print(name, param.data)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Display weights after training
print("\nWeights after training:")
for name, param in model.named_parameters():
    print(name, param.data)

# Evaluate model
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test).item()
    print(f'Test MSE: {mse:.4f}')