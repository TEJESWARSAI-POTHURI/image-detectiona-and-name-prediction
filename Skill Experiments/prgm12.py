import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# -------------------------------
# Hyperparameters
# -------------------------------
batch_size = 32
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Autoencoder Model
# -------------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), # 8x8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()  # 4x4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, output_padding=1, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# -------------------------------
# ResNet Model on Encoded Features
# -------------------------------
class ResNetOnEncoded(nn.Module):
    def __init__(self):
        super(ResNetOnEncoded, self).__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)

# -------------------------------
# Load Data (FakeData for testing)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_data = datasets.FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# -------------------------------
# Phase 1: Train Autoencoder
# -------------------------------
autoencoder = AutoEncoder().to(device)
ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("🔧 Training Autoencoder...")
for epoch in range(epochs):
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        _, decoded = autoencoder(images)
        loss = loss_fn(decoded, images)

        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: AE Loss = {total_loss/len(train_loader):.4f}")

# -------------------------------
# Phase 2: Train ResNet Classifier
# -------------------------------
print("\n🔧 Training ResNet on Encoded Images...")
resnet = ResNetOnEncoded().to(device)
resnet_optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
clf_loss_fn = nn.CrossEntropyLoss()

# Freeze autoencoder
for param in autoencoder.parameters():
    param.requires_grad = False

for epoch in range(epochs):
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        encoded, _ = autoencoder(images)

        outputs = resnet(encoded)
        loss = clf_loss_fn(outputs, labels)

        resnet_optimizer.zero_grad()
        loss.backward()
        resnet_optimizer.step()

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: ResNet Accuracy = {100 * correct / total:.2f}%")
