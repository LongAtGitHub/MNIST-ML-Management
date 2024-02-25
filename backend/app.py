from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys  
sys.path.append('../model')
from simple_model import SimpleMNISTModel  # Adjust the import path as necessary

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld():
  return "Backend Index"

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    learning_rate = float(data.get('learning_rate', 0.01))
    batch_size = int(data.get('batch_size', 64))
    epochs = int(data.get('epochs', 1))
    dropout_rate = float(data.get('dropout_rate', 0.0))  # Assuming dropout is handled inside your model
    print("done dropout")
    # Initialize model, device, loss, and optimizer
    model = SimpleMNISTModel(dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("done init")
    # Training dataset and loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("asdas")
    # Training loop
    model.train()
    print(epochs)
    for epoch in range(epochs):
        print("epoch", epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("training loop")
    # Test dataset and loader for validation
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("test")
    # Validation loop to calculate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print("validation")
    accuracy = 100 * correct / total
    print(accuracy)
    return jsonify({
        "message": "Training completed",
        "accuracy": accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
