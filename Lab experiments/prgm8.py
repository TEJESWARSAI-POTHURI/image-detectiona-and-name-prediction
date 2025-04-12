import os
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset URL and paths
dataset_url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
dataset_dir = "dataset/"
dataset_zip = dataset_dir + "electric_power.zip"
dataset_file = dataset_dir + "household_power_consumption.txt"

# Ensure dataset directory exists
os.makedirs(dataset_dir, exist_ok=True)

# Download dataset if not already downloaded
if not os.path.exists(dataset_file):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_zip)
    print("Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Dataset ready!")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(dataset_file, sep=";", low_memory=False, na_values=["?"], parse_dates=[[0, 1]], infer_datetime_format=True)

# Drop missing values
df.dropna(inplace=True)

# Convert 'Global_active_power' to float
df["Global_active_power"] = df["Global_active_power"].astype(float)

# Normalize the data
scaler = MinMaxScaler()
df["Global_active_power"] = scaler.fit_transform(df[["Global_active_power"]])

# Convert time series to sequences
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

seq_length = 24  # Predict next hour based on past 24 hours
data_seq, data_target = create_sequences(df["Global_active_power"].values, seq_length)

# Convert to PyTorch tensors
data_seq = torch.tensor(data_seq, dtype=torch.float32).to(device)
data_target = torch.tensor(data_target, dtype=torch.float32).to(device)

# Create DataLoader
train_size = int(0.8 * len(data_seq))
train_data, test_data = data_seq[:train_size], data_seq[train_size:]
train_labels, test_labels = data_target[:train_size], data_target[train_size:]

train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=64, shuffle=False)

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        return self.fc(hn[-1])

# Initialize model, loss, optimizer
model = RNNModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for seqs, labels in train_loader:
        seqs, labels = seqs.unsqueeze(-1), labels.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.6f}")

# Save model
torch.save(model.state_dict(), "rnn_energy_model.pth")

# Evaluate model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.unsqueeze(-1)
        outputs = model(seqs).cpu().numpy()
        predictions.extend(outputs.flatten())
        actuals.extend(labels.cpu().numpy())

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.plot(actuals[:100], label="Actual", linestyle="-")
plt.plot(predictions[:100], label="Predicted", linestyle="dashed")
plt.legend()
plt.title("Energy Consumption Prediction using RNN")
plt.show()

