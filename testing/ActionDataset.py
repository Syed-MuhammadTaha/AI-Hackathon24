import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import json

class ActionRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_json, action_txt_dir, transform=None):
        # Load the JSON file (annotations)
        with open(annotation_json, 'r') as f:
            annotation_json = json.load(f)  # This loads the file as a dictionary
        self.image_dir = image_dir
        self.annotations = annotation_json['annotations']
        self.images = annotation_json['images']
        self.action_txt_dir = action_txt_dir
        self.transform = transform

        # Create a mapping for easy lookup by image_id
        self.image_id_map = {image['id']: image for image in self.images}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_info = self.image_id_map[image_id]
        
        # Load the image using PIL
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')  # Use Pillow to load the image

        # Extract keypoints, bbox, and action labels (assumed action_id is part of annotation)
        keypoints = torch.tensor(annotation['keypoints'], dtype=torch.float32)
        bbox = torch.tensor(annotation['bbox'], dtype=torch.float32)
        label = torch.tensor(annotation['category_id'], dtype=torch.long)  # Assuming 'category_id' is the label

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, keypoints, label
