import torch
from torch import nn
from config import *
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
    def __init__(self, latent_size=latent_size, num_heads=num_heads, device=device, dropout=dropout):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        # Normalization layer for both sublayers
        self.norm = nn.LayerNorm(self.latent_size)
        
        # Multi-Head Attention layer
        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, dropout=self.dropout)          

        # MLP_head layer in the encoder. I use the same configuration as that 
        # used in the original VitTransformer implementation. The ViT-Base
        # variant uses MLP_head size 3072, which is latent_size*4.
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, embedded_patches):
        """
        Forward pass of the EncoderBlock module.

        Args:
            embedded_patches (torch.Tensor): The embedded input.

        Returns:
            torch.Tensor: The output of the second residual connection.
        """
        # First sublayer: Norm + Multi-Head Attention + residual connection.
        # We take the first element ([0]) of the returned output from nn.MultiheadAttention()
        # because this module returns 'Tuple[attention_output, attention_output_weights]'. 
        # Refer to here for more info: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html 
        firstNorm_out = self.norm(embedded_patches)
        attention_output = self.multihead(firstNorm_out, firstNorm_out, firstNorm_out)[0]

        # First residual connection
        first_added_output = attention_output + embedded_patches

        # Second sublayer: Norm + enc_MLP (Feed forward)
        secondNorm_out = self.norm(first_added_output)
        ff_output = self.enc_MLP(secondNorm_out)

        # Return the output of the second residual connection
        return ff_output + first_added_output