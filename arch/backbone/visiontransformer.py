import torch
import torch.nn as nn
from transformers import ViTModel

__all__ = ['VisionTransformer']

class VisionTransformer(nn.Module):
    def __init__(self, variant="vit-base-patch16-224", return_idx=[0, 1, 2, 3]):
        super(VisionTransformer, self).__init__()
        
        self.variant_dict = {
            "vit-base-patch16-224": "google/vit-base-patch16-224",
            "vit-large-patch16-224": "google/vit-large-patch16-224",
            "vit-base-patch32-384": "google/vit-base-patch32-384",
            "vit-large-patch32-384": "google/vit-large-patch32-384",
        }

        if variant not in self.variant_dict:
            raise ValueError(f"Variant '{variant}' is not a valid Vision Transformer model. Please choose from: {list(self.variant_dict.keys())}")
        
        self.model = ViTModel.from_pretrained(self.variant_dict[variant])
        
        self.return_idx = return_idx

    def forward(self, x):
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Validate return_idx
        if max(self.return_idx) >= len(hidden_states):
            raise ValueError(f"Invalid return_idx {self.return_idx}. `hidden_states` has {len(hidden_states)} layers.")
        
        # Extract selected hidden states
        selected_states = [hidden_states[idx] for idx in self.return_idx]
        return selected_states

