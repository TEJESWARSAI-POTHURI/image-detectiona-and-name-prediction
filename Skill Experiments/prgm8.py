import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data Augmentation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# 2. Load dataset (you can replace FakeData with your real dataset)
dataset = datasets.FakeData(transform=transform, size=1000)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. CNN Architecture with Regularization
class AugmentedCNN(nn.Module):
    def __init__(self):
        super(AugmentedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 -> 4
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. Initialize model, loss, optimizer
model = AugmentedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 reg

# 5. Train just 1 batch (demo)
model.train()
for images, labels in loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # Remove break to train fully

# 6. Hook to capture intermediate layer outputs
activations = {}
def get_activation(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

# Register hooks for conv1, conv2, conv3
model.conv1.register_forward_hook(get_activation("conv1"))
model.conv2.register_forward_hook(get_activation("conv2"))
model.conv3.register_forward_hook(get_activation("conv3"))

# 7. Forward pass one batch for visualization
model.eval()
images, labels = next(iter(loader))
_ = model(images)

# 8. Show activations
def show_activations(activation, title):
    act = activation[0].detach()  # First image in batch
    fig, axs = plt.subplots(1, min(8, act.size(0)), figsize=(15, 5))
    fig.suptitle(title)
    for i in range(min(8, act.size(0))):
        axs[i].imshow(act[i], cmap='gray')
        axs[i].axis('off')
    plt.show()

# Visualize layer-wise outputs
show_activations(activations["conv1"], "Conv1 Activations")
show_activations(activations["conv2"], "Conv2 Activations")
show_activations(activations["conv3"], "Conv3 Activations")
