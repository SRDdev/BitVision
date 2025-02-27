"""
This script trains a Vision Transformer model on the CIFAR-10 dataset.
It includes configuration loading, dataset preparation, model initialization,
training loop, and evaluation. The training process is logged with detailed
information about the progress and performance metrics.
"""
import torch
from torch import nn, optim
from tqdm import tqdm
import time
import logging
from pathlib import Path
from colorama import Fore, Style, init
from Bit.dataset import DatasetLoader
from Bit.visionTransformer import VisionTransformer
from config.config import load_config

# Initialize colorama
init(autoreset=True)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Load configuration
config = load_config("config/config.yaml")
device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

# Training parameters
epochs = config.get("epochs", 10)
batch_size = config.get("batch_size", 32)
learning_rate = config.get("base_lr", 3e-4)
weight_decay = config.get("weight_decay", 1e-4)
dataset_size = config.get("dataset_size", 0.1)  # Use 10% of data by default

# Define criterion globally
criterion = nn.CrossEntropyLoss()

# Ensure checkpoint directory exists
Path("checkpoints").mkdir(exist_ok=True)

def print_gpu_info():
    if torch.cuda.is_available():
        logger.info(f"{Fore.GREEN}Using GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}GPU Memory Usage:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        logger.info(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.RED}GPU not available, using CPU{Style.RESET_ALL}")

@torch.no_grad()
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(data_loader), 100 * correct / total

def train():
    # Print system info
    logger.info(f"{Fore.MAGENTA}=== Training Configuration ==={Style.RESET_ALL}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Weight Decay: {weight_decay}")
    logger.info(f"Dataset Size: {dataset_size*100}% of full dataset")
    logger.info(f"Device: {device}")
    print_gpu_info()
    
    # Initialize dataset loader
    logger.info(f"{Fore.BLUE}Loading datasets...{Style.RESET_ALL}")
    dataset_loader = DatasetLoader(
        dataset_size=dataset_size,
        batch_size=batch_size,
        num_workers=2
    )
    train_loader, test_loader = dataset_loader.load_cifar10()
    
    # Initialize model
    logger.info(f"{Fore.CYAN}Initializing model...{Style.RESET_ALL}")
    model = VisionTransformer(
        num_encoders=config.get("num_encoders", 12),
        latent_size=config.get("latent_size", 768),
        device=device,
        num_classes=config.get("num_classes", 10)
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"{Fore.GREEN}Using {torch.cuda.device_count()} GPUs!{Style.RESET_ALL}")
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    train_start_time = time.time()
    
    logger.info(f"{Fore.MAGENTA}=== Starting Training ==={Style.RESET_ALL}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        progress_bar = tqdm(train_loader, desc=f"{Fore.YELLOW}Epoch {epoch+1}/{epochs}{Style.RESET_ALL}", leave=True)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{Fore.GREEN}{100 * correct / total:.2f}%{Style.RESET_ALL}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Log epoch results
        logger.info(f"{Fore.BLUE}Epoch {epoch+1}/{epochs}{Style.RESET_ALL} - "
                    f"Train Loss: {total_loss / len(train_loader):.4f} - "
                    f"Train Acc: {100 * correct / total:.2f}% - "
                    f"Test Loss: {test_loss:.4f} - "
                    f"Test Acc: {Fore.GREEN}{test_acc:.2f}%{Style.RESET_ALL}")
    
    total_time = time.time() - train_start_time
    logger.info(f"{Fore.MAGENTA}=== Training Complete ==={Style.RESET_ALL}")
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Best test accuracy: {Fore.GREEN}{best_acc:.2f}%{Style.RESET_ALL}")

if __name__ == "__main__":
    train()
