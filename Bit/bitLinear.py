"""
BitLinear Layer with Straight-Through Estimator (STE) based Quantization.
Improved version combining ideas from BitNet paper and Huggingface blog.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------------- #
# Custom autograd functions for STE
# -------------------------------------------------------------------------------------------------- #
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# -------------------------------------------------------------------------------------------------- #
# Quantization Functions
# -------------------------------------------------------------------------------------------------- #
def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token quantization to 8 bits using STE for rounding.

    Args:
        x (Tensor): Input activation.

    Returns:
        Tensor: Quantized activation tensor.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=torch.finfo(x.dtype).eps)
    x_scaled = x * scale
    x_quant = RoundSTE.apply(x_scaled).clamp(-128, 127) / scale
    return x_quant

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Weight quantization to ternary (-1, 0, 1) with STE.

    Args:
        w (Tensor): Weights.

    Returns:
        Tensor: Quantized weights.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_scaled = w * scale
    w_quant = RoundSTE.apply(w_scaled).clamp(-1, 1) / scale
    return w_quant

# -------------------------------------------------------------------------------------------------- #
# BitLinear Layer
# -------------------------------------------------------------------------------------------------- #
class BitLinear(nn.Linear):
    """
    Linear layer with activation and weight quantization using STE.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.rms_norm = nn.LayerNorm(in_features, elementwise_affine=False)  # RMSNorm is approx LayerNorm without affine params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization applied to inputs and weights.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Normalize input
        x_norm = self.rms_norm(x)

        # STE quantization trick: use detached delta
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # Perform quantized linear operation
        return F.linear(x_quant, w_quant, self.bias)
