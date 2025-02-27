"""
This file contains the implementation of the EncoderBlock module for the Vision Transformer (ViT).
"""
import torch
from torch import nn
from Bit.bitLinear import BitLinear
#--------------------------------Encoder---------------------------#
class EncoderBlock(nn.Module):
    """
    Defines the EncoderBlock module for the Vision Transformer (ViT).

    Args:
        latent_size (int): size of the latent space.
        num_heads (int): number of attention heads.
        device (torch.device): device (cpu, cuda) on which the model is run.
        dropout (float): dropout rate.

    Outputs:
        torch.Tensor: The embedded input.

    Note:
        This module is used to process the embedded input through a series of sublayers,
        including normalization, multi-head attention, and a feed-forward network.
        Each sublayer is followed by a residual connection.
    """
    def __init__(self, latent_size: int = 256, num_heads: int = 8, device: str = "cpu", dropout: float = 0.5):
        super(EncoderBlock, self).__init__()
        
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        # Normalization
        self.norm = nn.LayerNorm(self.latent_size)

        # Multi-Headed Attention Layer
        self.multihead = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)

        # MLP_head layer in the encoder. I use the same configuration as that used in the original VitTransformer implementation. The ViT-Base variant uses MLP_head size 3072, which is latent_size*4.

        self.enc_MLP = nn.Sequential(
            BitLinear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            BitLinear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )
        self.to(self.device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderBlock module.

        Args:
            input_tensor (torch.Tensor): The embedded input tensor.

        Returns:
            torch.Tensor: The output after applying multi-head attention and MLP with residual connections.
        """
        normalized_input = self.norm(input_tensor)
        attention_output = self.multihead(normalized_input, normalized_input, normalized_input)[0]
        residual_attention = input_tensor + attention_output
        normalized_residual = self.norm(residual_attention)
        mlp_output = self.enc_MLP(normalized_residual)
        final_output = residual_attention + mlp_output

        return final_output

