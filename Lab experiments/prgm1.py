import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
torch.manual_seed(42)

num_samples = 1000
num_features = 5

# Features: Age, Income, Credit Utilization, Late Payments, Debt-to-Income Ratio
X = np.random.rand(num_samples, num_features) * 100  # Random values scaled

# Target: 1 for high risk, 0 for low risk (synthetic thresholding)
y = (X[:, 1] / X[:, 4] > 1.5).astype(int)

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test,
                                                                                       dtype=torch.float32).view(-1, 1)


# Define the neural network model
class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super(CreditRiskNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Initialize model, loss, and optimizer
model = CreditRiskNN(num_features)
criterion = nn.BCELoss()
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
    y_pred_class = (y_pred >= 0.5).float()
    accuracy = (y_pred_class.eq(y_test).sum() / y_test.shape[0]).item()
    print(f'Accuracy: {accuracy:.4f}')