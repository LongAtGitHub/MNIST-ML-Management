from flask import Flask, request, jsonify
# from simple_model import SimpleMNISTModel
from flask_cors import CORS
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld():
  return "Backend Index"
  
# Assume model training and evaluation happens here
@app.route('/train', methods=['POST'])
def train_model():
    # Example of receiving hyperparameters
    data = request.json
    learning_rate = data.get('learning_rate', 0.01)
    batch_size = data.get('batch_size', 64)
    epochs = data.get('epochs', 10)
    dropout_rate = data.get('dropout_rate', 0.5)
    # Placeholder for actual training logic
    # You should integrate the training loop here
    # print('world')
    return jsonify({"message": "Training started with batch size {}, epochs {}, dropout rate {}, and learning rate {}".format(batch_size, epochs, dropout_rate, learning_rate)})


if __name__ == '__main__':
    app.run(debug=True)
