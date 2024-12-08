import torch
import torch.nn as nn
from torchvision import models

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes, vit_model):
        super(ActionRecognitionModel, self).__init__()
        
        # Use a pre-trained Vision Transformer model
        self.vit = vit_model
        
        # Forward pass through ViT (without the classification head)
        vit_output = self.vit(torch.zeros(1, 3, 224, 224))  # Pass a dummy tensor to get the output shape
        vit_features = vit_output.shape[1]  # Get the output feature size (e.g., 768 for ViT-B-16)
        
        # Define the fully connected layer
        self.fc = nn.Linear(vit_features + 42, num_classes)  # 42 is the flattened keypoints size (21 keypoints * 2)

    def forward(self, x, keypoints):
        # Extract features from ViT
        vit_features = self.vit(x)
        
        # Flatten ViT features
        vit_features = vit_features.flatten(start_dim=1)
        
        # Concatenate ViT features and keypoints
        combined_features = torch.cat((vit_features, keypoints), dim=1)
        
        # Classification head
        logits = self.fc(combined_features)
        return logits
