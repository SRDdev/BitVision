# BitVision

BitVision is an implementation of Vision Transformers (ViT) with bit-wise quantization, based on the BitNet architecture. This project provides an efficient approach to vision transformers by using binary quantization techniques while maintaining model performance.

## Features

- Vision Transformer (ViT) implementation with BitNet quantization
- 8-bit activation quantization
- Binary (-1, 1) weight quantization
- CIFAR-10 dataset support with configurable dataset size
- Flexible configuration system
- Comprehensive training pipeline with logging
- GPU support with multi-GPU capability

## Project Structure
```
BitVision/
├── Bit/
│   ├── bitLinear.py          # BitNet linear layer implementation
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── embeddings.py         # Input embedding for ViT
│   ├── encoder.py            # Transformer encoder implementation
│   └── visionTransformer.py  # Main ViT model
├── config/
│   ├── config.py             # Configuration loader
│   └── config.yaml           # Model and training configurations
└── train.py                  # Training script
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BitVision.git
   cd BitVision
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm colorama pyyaml
   ```

## Configuration

The model and training parameters can be configured in `config/config.yaml`. Key configurations include:

- **Model architecture parameters** (number of encoders, latent size)
- **Training parameters** (batch size, learning rate, epochs)
- **Dataset settings**
- **Device settings** (CPU/GPU)
- **Learning rate scheduler settings**

## Usage

To train the model, run:
   ```bash
   python train.py
   ```

### The training script will:
- Load and preprocess the CIFAR-10 dataset
- Initialize the Vision Transformer model
- Train the model with the specified configuration
- Log training progress and metrics
- Save model checkpoints

## Technical Details

### Bit Linear Layer
- Implements activation quantization to 8 bits
- Weight quantization to binary values (-1, 1)
- Uses RMSNorm for input normalization

### Vision Transformer Architecture
- Patch-based image embedding
- Positional embeddings
- Multi-head self-attention
- BitLinear transformations
- Classification head

### Training Features
- Adam optimizer with weight decay
- Learning rate scheduling
- Multi-GPU support
- Progress tracking with tqdm
- Colored logging output

## Performance Monitoring

The training process provides detailed monitoring:
- Training/validation loss
- Accuracy metrics
- GPU memory usage
- Learning rate adjustments
- Training time statistics

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA (optional, for GPU support)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM recommended

## License
[Add your license information here]

## Contributing
[Add contribution guidelines here]

## Acknowledgments
This implementation is based on the BitNet paper: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/pdf/2310.11453).

