#Reference: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import os
import json
import io
from PIL import Image
import logging

MODEL_FILE_NAME = 'model.pth'
NUM_CLASSES = 133

def Net():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, NUM_CLASSES))
    return model

def model_fn(model_dir):
    model = Net()
    logging.debug("Model created")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    with open(os.path.join(model_dir, MODEL_FILE_NAME), "rb") as f:
        model.load_state_dict(torch.load(f, map_location = device))
    logging.debug("Model loaded") 
    model.to(device).eval()
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type=='image/jpeg'
    return Image.open(io.BytesIO(request_body))


def predict_fn(input_data, model):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_data = test_transform(input_data).unsqueeze(0).to(device)
        predictions = model(input_data)
    return predictions


def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)