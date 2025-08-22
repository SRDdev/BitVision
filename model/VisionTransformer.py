import torch
from torch import nn
from config import *
from .Embedding import InputEmbedding
from .Encoder import EncoderBlock
#----------------------------Vision Transformer--------------------------#
class VisionTransformer(nn.Module):
    """
    Defines the VitTransformer module for the Vision Transformer (ViT) architecture.

    Args:
        num_encoders (int): number of encoder layers.
        latent_size (int): size of the latent space.
        device (torch.device): device (cpu, cuda) on which the model is run.
        num_classes (int): number of classes for classification.
        dropout (float): dropout rate.

    Outputs:
        torch.Tensor: The classification vector for all images in the batch.

    Note:
        This module is the main component of the ViT architecture,
        consisting of an input embedding, a stack of encoder layers, and an MLP head
        for classification.
    """
    def __init__(self, num_encoders=num_encoders, latent_size=latent_size, device=device, num_classes=num_classes, dropout=dropout):
        super(VisionTransformer, self).__init__()
        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.embedding = InputEmbedding()

        # Create a stack of encoder layers
        self.encStack = nn.ModuleList([EncoderBlock() for i in range(self.num_encoders)])

        # MLP_head at the classification stage has 'one hidden layer at pre-training time
        # and by a single linear layer at fine-tuning time'. For this implementation I will
        # use what was used for training, so I'll have a total of two layers, one hidden
        # layer and one output layer.
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
        )

    def forward(self, test_input):
        """
        Forward pass of the VitTransformer module.

        Args:
            test_input (torch.Tensor): input image passed to the model.

        Returns:
            torch.Tensor: The classification vector for all images in the batch.
        """
        # Apply input embedding (patchify + linear projection + position embeding)
        # to the input image passed to the model
        enc_output = self.embedding(test_input)

        # Loop through all the encoder layers
        for enc_layer in self.encStack:
            enc_output = enc_layer.forward(enc_output)

        # Extract the output embedding information of the [class] token
        cls_token_embedding = enc_output[:, 0]

        # Finally, return the classification vector for all image in the batch
        return self.MLP_head(cls_token_embedding)