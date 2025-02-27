"""
This module defines the InputEmbedding class for the Vision Transformer (ViT) model.
The InputEmbedding class is responsible for embedding input images into a sequence of fixed-sized patches,
which are then fed into a Transformer model. The class uses a custom BitLinear layer for the linear projection
of the image patches and includes a class token and positional embedding.
"""

import torch 
from torch import nn
import einops
from Bit.bitLinear import BitLinear
#--------------------------------Input Embedding--------------------------------#
class InputEmbedding(nn.Module):
    """
    Defines the InputEmbedding module for the Vision Transformer (ViT).

    Args:
        patch_size (int): size of the image patches.
        n_channels (int): number of channels in the input image.
        device (torch.device): device (cpu, cuda) on which the model is run.
        latent_size (int): size of the latent space.
        batch_size (int): batch size.

    Outputs:
        torch.Tensor: The embedded input.

    Note:
        This module is used to embed input images into a sequence of fixed-sized patches,
        which are then fed into a Transformer model.
    """
    def __init__(self, patch_size: int = 16, n_channels: int = 3, device: str = "cpu", latent_size: int = 256, batch_size: int = 8):
        super(InputEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.latent_size = latent_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Replace the nn.Linear Projection Layer from PyTorch with BitLinear
        self.bitlinearProjection = BitLinear(self.input_size, self.latent_size)

        # Random initialization of of [class] token that is prepended to the linear projection vector.
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.latent_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InputEmbedding module.

        Args:
            input_data (torch.Tensor): input image.

        Returns:
            torch.Tensor: The embedded input.
        """
        x = x.to(self.device)
        patches = einops.rearrange(x, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        bit_linear_projection = self.bitlinearProjection(patches)
        b, n, _ = bit_linear_projection.shape
        class_token = self.class_token.expand(b, -1, -1) 
        bit_linear_projection = torch.cat([class_token, bit_linear_projection], dim=1)
        pos_embed = self.pos_embedding.expand(b, n + 1, -1)
        bit_linear_projection += pos_embed
        return bit_linear_projection
    


