import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from numpy import linalg as LA

class VGGNet:
    def __init__(self):
        # Load the pre-trained VGG16 model from torchvision
        self.model = models.vgg16(pretrained=True).features
        self.model.eval()  # Set the model to evaluation mode (no backpropagation)
        
        # Define the preprocessing transformations (resize, normalization)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])

    def extract_feat(self, img_path):
        # Load image using PIL
        img = Image.open(img_path).convert('RGB')  # Ensure 3-channel RGB
        
        # Preprocess image
        img = self.preprocess(img).unsqueeze(0)  # Add batch dimension

        # Perform feature extraction
        with torch.no_grad():  # Disable gradient computation
            feat = self.model(img)
            feat = feat.view(feat.size(0), -1)  # Flatten features
        
        # Normalize feature vector
        norm_feat = feat[0].numpy() / LA.norm(feat[0].numpy())
        return norm_feat

