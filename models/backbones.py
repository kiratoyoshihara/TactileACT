import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    ResNet backbone to extract spatial features from images.
    It removes the final FC and Global Average Pooling layers.
    """
    def __init__(self, hidden_dim, pretrained=True):
        super().__init__()
        # Load ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Remove the last two layers (avgpool and fc)
        # We want the feature map, e.g., [B, 512, 7, 7]
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        
        # Linear projection to match Transformer hidden dimension
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

    def forward(self, x):
        """
        Input: [B, 3, H, W]
        Output: [B, Num_Patches, Hidden_Dim] sequence of tokens
        """
        # Extract features [B, 512, 7, 7] (for 224x224 input)
        features = self.body(x)
        
        # Project channel dimension [B, Hidden_Dim, 7, 7]
        features = self.input_proj(features)
        
        # Flatten spatial dimensions to create a sequence
        # [B, Hidden_Dim, 49] -> [B, 49, Hidden_Dim]
        return features.flatten(2).permute(0, 2, 1)