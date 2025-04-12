import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from torchvision import models



# Function to load and preprocess test data
def load_test_data(data_dir, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader, dataset.classes, dataset


# Function to predict and display random test images with predictions
def predict_random_images(model, test_dataset, class_names, device, num_images=5):
    model.eval()
    random_indices = random.sample(range(len(test_dataset)), num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i, idx in enumerate(random_indices):
        image, label = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_names[predicted.item()]
            actual_label = class_names[label]

        # De-normalize image for visualization
        image = image.numpy().transpose(1, 2, 0)
        image = (image * 0.5) + 0.5  # De-normalize to [0, 1]
        axes[i].imshow(np.clip(image, 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")

    plt.show()


# Main function
def main():
    model_path = "D:/deep learning/project/trained_model.pth"
    test_data_dir = "D:/deep learning/Vehicles/Test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_loader, class_names, test_dataset = load_test_data(test_data_dir)
    print("Class names:", class_names)

    # Load trained model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Predict and display random images
    predict_random_images(model, test_dataset, class_names, device, num_images=5)


if __name__ == "__main__":
    main()
