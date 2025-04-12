from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
import os
import tempfile
from training import build_model, predict

app = Flask(__name__)

# Initialize model and load trained weights
model_path = "D:/DL/project/trained_model.pth"  # Path to the saved trained model
num_classes = 7  # Update this based on your dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build and load the model
model = build_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Class names
class_names = ['Auto Rickshaws', 'Bikes', 'Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains']

# Define the prediction function (integrated with Flask)
def predict_flask(model, image_path, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Get the uploaded file
        image_file = request.files['image']

        if image_file:
            # Save the image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image_path = temp_file.name
                image_file.save(image_path)

            try:
                # Get the prediction
                result = predict_flask(model, image_path, class_names=class_names, device=device)
            finally:
                # Delete the temporary file
                os.remove(image_path)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
