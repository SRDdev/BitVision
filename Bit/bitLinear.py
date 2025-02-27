"""
This file contains the implementation of Bit Linear Layer from the paper bit net paper (https://arxiv.org/pdf/2310.11453)
"""
import torch
from torch import nn
from torch.nn.modules.normalization import RMSNorm

# -------------------------------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------------------------------- #
def activation_quant(x: torch.Tensor):
    """
    Per token quantization to 8bits. No grouping is needed for quantization
    Args:
        x (Tensor): Input Activation

    Returns:
        y (Tensor): Quantized Activation in 8bit.
    """
    scale = (2**7) / x.abs().max(dim=-1, keepdim=True).values.clamp(min=torch.finfo(x.dtype).eps)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weights_quant(w: torch.Tensor):
    """
    Quantizing the weights to zero mean and then converting them to -1,1.
    Args:
        w (Tensor): Weights
    Returns:
        u (Tensor): Quantized Weights with either 1,-1
    """
    scale = w.std().clamp(min=torch.finfo(w.dtype).eps)
    u = (w - w.mean()).sign() * scale
    return u
# -------------------------------------------------------------------------------------------------- #
# Bit Linear Layer
# -------------------------------------------------------------------------------------------------- #
class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.rms_norm = RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_norm = self.rms_norm(x) 
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = self.weight + (weights_quant(self.weight) - self.weight).detach()
        return nn.functional.linear(x_quant, w_quant)