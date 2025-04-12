import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Hyperparameters
input_size = 32      # Each row of image treated as sequence input
hidden_size = 64
num_layers = 1
num_classes = 10
batch_size = 32
seq_length = 32      # 32 rows of image
epochs = 3

# 2. Define RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, channels, height, width] => we treat height as time steps
        x = x.squeeze(1)         # [B, 1, H, W] → [B, H, W]
        out, _ = self.rnn(x)     # [B, H, hidden]
        out = self.fc(out[:, -1, :])  # Last time step
        return out

# 3. Dataset & Dataloader
transform = transforms.Compose([
    transforms.Grayscale(),           # RNN works better with 1 channel
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = datasets.FakeData(size=1000, image_size=(1, 32, 32), transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
for epoch in range(epochs):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}, Accuracy: {100 * correct / total:.2f}%")
