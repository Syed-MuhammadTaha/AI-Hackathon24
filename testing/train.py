import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report
from ActionRecognition import ActionRecognitionModel
from ActionDataset import ActionRecognitionDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
num_classes = 5  # Set the number of action classes
num_epochs = 10  # Number of epochs
batch_size = 32  # Batch size
learning_rate = 1e-4  # Learning rate

# Load the Vision Transformer (ViT) model (pretrained)
vit_model = models.vit_b_16(pretrained=True)  # Load ViT model
vit_model.head = nn.Identity()  # Remove the classification head if you don't need it
model = ActionRecognitionModel(num_classes=5, vit_model=vit_model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the dataset and dataloaders
dataset = ActionRecognitionDataset(
    image_dir='/home/user/AI-Hackathon24/data/xml_labels/FRAMES',  # Directory containing your image frames
    annotation_json='/home/user/AI-Hackathon24/data/merged.json',  # Path to your annotation JSON
    action_txt_dir='/home/user/AI-Hackathon24/data/actions',  # Directory containing the action txt files
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, keypoints, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, keypoints)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    # Save model checkpoint every few epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'action_recognition_epoch_{epoch + 1}.pth')
