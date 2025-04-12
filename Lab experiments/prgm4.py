import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data (100 samples, 4 features)
num_samples = 100
num_features = 4
X = np.random.rand(num_samples, num_features) * 10  # Values between 0 and 10

y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] - 0.5 * X[:, 3] + np.random.randn(
    num_samples) * 2  # Linear relation + noise

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test,
                                                                                       dtype=torch.float32).view(-1, 1)


# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation in output layer (Regression problem)
        return x


# Initialize the model
input_size = 4
hidden_size = 2
output_size = 1
model = SimpleNN(input_size, hidden_size, output_size)

# Define loss function and optimizer (SGD)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with Stochastic Gradient Descent
epochs = 500
batch_size = 10
num_batches = len(X_train) // batch_size

for epoch in range(epochs):
    for i in range(num_batches):
        # Get batch indices
        start = i * batch_size
        end = start + batch_size

        # Select mini-batch
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate on test data
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test).item()
    print(f'Final Test MSE: {mse:.4f}')