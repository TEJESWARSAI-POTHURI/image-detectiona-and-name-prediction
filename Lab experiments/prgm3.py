import torch
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic input data (100 samples, 4 features)
num_samples = 100
num_features = 4
X = np.random.rand(num_samples, num_features) * 10  # Values between 0 and 10

y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] - 0.5 * X[:, 3] + np.random.randn(
    num_samples) * 2  # Linear relation + noise

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define network structure
input_size = 4
hidden_size = 2
output_size = 1

# Initialize weights and biases manually
W1 = torch.randn(input_size, hidden_size, dtype=torch.float32, requires_grad=True)  # (4x2)
b1 = torch.randn(hidden_size, dtype=torch.float32, requires_grad=True)  # (2,)
W2 = torch.randn(hidden_size, output_size, dtype=torch.float32, requires_grad=True)  # (2x1)
b2 = torch.randn(output_size, dtype=torch.float32, requires_grad=True)  # (1,)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Training loop (Manual Gradient Descent)
for epoch in range(epochs):
    # Forward pass
    Z1 = X @ W1 + b1  # Linear transformation
    A1 = torch.relu(Z1)  # Activation function
    Z2 = A1 @ W2 + b2  # Output layer

    loss = torch.mean((Z2 - y) ** 2)  # Mean Squared Error Loss

    # Backpropagation
    loss.backward()

    # Gradient Descent Update Rule (Manual weight update)
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

    # Zero gradients after updating
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Final Weights and Biases
print("\nFinal Weights and Biases:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)