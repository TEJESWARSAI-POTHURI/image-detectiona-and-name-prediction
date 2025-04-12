from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)


# Define Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Load trained model
model = SimpleNN()
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
model.eval()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values
    input1 = float(request.form['input1'])
    input2 = float(request.form['input2'])

    # Convert to tensor
    inputs = torch.tensor([[input1, input2]], dtype=torch.float32)

    # Get prediction
    output = model(inputs).item()

    return render_template('index.html', prediction=f'Prediction: {output:.4f}')


if __name__ == '__main__':
    app.run(debug=True)
