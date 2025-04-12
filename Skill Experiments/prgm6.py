import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import random

# 1. Load Dummy Dataset (FakeData used here)
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FakeData(transform=transform, size=1000)

# 2. Load Predefined Model (ResNet18 for example)
model = models.resnet18(num_classes=10)  # Change classes if needed
model.eval()  # Set to evaluation mode

# 3. Mini-batch Evaluation Function
def evaluate_random_minibatches(model, dataset, batch_size=64, num_batches=5, device='cpu'):
    model.to(device)
    model.eval()
    total_accuracy = 0.0

    for _ in range(num_batches):
        indices = random.sample(range(len(dataset)), batch_size)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size)

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total_accuracy += correct / labels.size(0)

    avg_accuracy = total_accuracy / num_batches
    print(f'Average Accuracy over {num_batches} random mini-batches: {avg_accuracy * 100:.2f}%')
    return avg_accuracy

# 4. Call the function
evaluate_random_minibatches(model, test_dataset, batch_size=64, num_batches=5)
