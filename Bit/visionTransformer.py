"""
visionTransformer.py

This module defines the Vision Transformer (ViT) model, a type of neural network architecture for image classification tasks.

The Vision Transformer model is implemented using PyTorch and consists of the following components:
- Input Embedding: Converts input images into a sequence of embeddings.
- Encoder Stack: A stack of encoder layers, each containing self-attention and feed-forward neural network components.
- MLP Head: A multi-layer perceptron for classification.

The model configuration is loaded from a YAML file, and the model can be run on either CPU or GPU.

Classes:
    VisionTransformer: Defines the Vision Transformer model.

Functions:
    forward: Defines the forward pass of the Vision Transformer model.
"""
import os
import sys
import torch
from torch import nn

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from config.config import load_config
from Bit.embeddings import InputEmbedding
from Bit.encoder import EncoderBlock
from Bit.bitLinear import BitLinear

# Load configuration
config = load_config("config/config.yaml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------- Vision Transformer ---------------------------- #
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) Model.

    Args:
        num_encoders (int): Number of encoder layers.
        latent_size (int): Latent feature dimension.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_classes (int): Number of classification classes.
        dropout (float): Dropout probability.

    Returns:
        torch.Tensor: Logits for each image in the batch.
    """

    def __init__(self, num_encoders: int = 2, latent_size: int = 768, device: str = 'cpu', num_classes: int = 2, dropout: float = 0.5):
        super(VisionTransformer, self).__init__()

        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.dropout = dropout

        self.input_embedding = InputEmbedding(latent_size=self.latent_size, device=self.device).to(self.device)
        self.encoder_stack = nn.ModuleList(
            [EncoderBlock(latent_size=self.latent_size, device=self.device) for _ in range(self.num_encoders)]
        ).to(self.device)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            BitLinear(self.latent_size, self.latent_size),
            nn.GELU(),
            BitLinear(self.latent_size, self.num_classes)
        ).to(self.device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            input_tensor (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Logits for each image in the batch.
        """
        input_tensor = input_tensor.to(self.device)
        encoder_output = self.input_embedding(input_tensor)
        for encoder_layer in self.encoder_stack:
            encoder_output = encoder_layer(encoder_output) 
        cls_token_embedding = encoder_output[:, 0]

        return self.mlp_head(cls_token_embedding).to(self.device)
