import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)   # 3 input channels -> 8 output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)  # Assuming 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.act1 = x                 # Save activation of conv1
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        self.act2 = x                 # Save activation of conv2
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)     # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Load dummy dataset (replace with your real dataset)
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
dataset = datasets.FakeData(transform=transform, size=1000)  # Replace FakeData with your dataset
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Step 3: Initialize model and get one sample
model = SimpleCNN()
model.eval()

images, labels = next(iter(loader))
output = model(images)

# Step 4: Visualize layer-wise outputs
def show_activations(activations, title):
    act = activations[0].detach()  # Get first image's activations
    num_channels = act.shape[0]

    fig, axs = plt.subplots(1, min(num_channels, 8), figsize=(15, 5))
    fig.suptitle(title)
    for i in range(min(num_channels, 8)):
        axs[i].imshow(act[i], cmap='gray')
        axs[i].axis('off')
    plt.show()

# Visualize conv1 and conv2 outputs
show_activations(model.act1, "Conv1 Activations")
show_activations(model.act2, "Conv2 Activations")
