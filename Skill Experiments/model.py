import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create dataset
X = torch.rand((100, 2))  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).float().view(-1, 1)  # Simple classification

# Initialize model, loss, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Save trained model
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved as trained_model.pth")
