import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Function to load and preprocess data
def load_data(data_dir, batch_size=32, augment=True):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=15, shear=10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, dataset.classes

# Function to visualize a batch of images
def visualize_data(data_loader, classes, num_samples=5):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images.numpy().transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    images = (images * 0.5) + 0.5  # De-normalize to [0, 1]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
    plt.show()

# Pretrained ResNet model setup
def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    train_losses,train_accuracies,val_losses, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0  # Initialize here

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # return model
    return train_losses, train_accuracies, val_losses, val_accuracies

# Function to evaluate and display confusion matrix
def evaluate_model(model, test_loader, classes, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=classes))

# Prediction function
def predict(model, image_path, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]
    return predicted_label


import random


# Function to evaluate on random mini-batches
def evaluate_random_mini_batches(model, test_loader, classes, device, num_batches=5):
    model.eval()
    random_batches = random.sample(list(test_loader), num_batches)

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in random_batches:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / num_batches
    accuracy = (total_correct / total_samples) * 100

    print(f"Random Mini-batch Evaluation ({num_batches} batches):")
    print(f"  - Average Loss: {avg_loss:.4f}")
    print(f"  - Accuracy: {accuracy:.2f}%")


# Main function
def main():
    data_dir_train = "D:/deep learning/Vehicles/Train"
    data_dir_test = "D:/deep learning/Vehicles/Test"

    train_loader, class_names = load_data(data_dir_train, batch_size=32, augment=True)
    test_loader, _ = load_data(data_dir_test, batch_size=32, augment=False)

    print("Class names:", class_names)

    visualize_data(train_loader, class_names)

    num_classes = len(class_names)
    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizers = {
        "Adam": optim.Adam(model.parameters(), lr=0.001),
        "SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "RMSprop": optim.RMSprop(model.parameters(),lr = 0.001)
        }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=20)

    num_epochs = 1
    for opt_name, optimizer in optimizers.items():
        print(f"\nTraining with {opt_name} optimizer")
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, test_loader, criterion, optimizer, device, num_epochs=num_epochs
        )

        print(f"\nResults for {opt_name} optimizer:")
        for epoch in range(num_epochs):
            print(
                f"Epoch {epoch + 1}: Train Loss = {train_losses[epoch]:.4f}, Train Accuracy = {train_accuracies[epoch]:.4f}, "
                f"Val Loss = {val_losses[epoch]:.4f}, Val Accuracy = {val_accuracies[epoch]:.4f}")

    # torch.save(trained_model.state_dict(), "D:/deep learning/skill/project/trained_model.pth")
    # evaluate_model(trained_model, test_loader, class_names, device)

    image_path = "/Vehicles/Test/Auto Rickshaws/Auto Rickshaw (726).jpg"
    # prediction = predict(trained_model, image_path, class_names, device)
    # print(f"The predicted label is: {prediction}")
    evaluate_random_mini_batches(model, test_loader, class_names, device, num_batches=5)
if __name__ == "__main__":
    main()
