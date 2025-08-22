import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from config import *
from data.dataset import dataset_builder
from model.VisionTransformer import VisionTransformer
import einops
from torchsummary import summary
import time
epochs = 10
#--------------Dataset----------#
trainloader = dataset_builder()
#--------------Model------------#
model = VisionTransformer(num_encoders, latent_size, device, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.LinearLR(optimizer)

#-----------------Train----------------#
def trainer():
    model.train().to(device)

    for epoch in tqdm(range(epochs), total=epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print('Batch {} epoch {} has loss = {}'.format(batch_idx, epoch, running_loss/200))                
                running_loss = 0

        scheduler.step()

if __name__ == "__main__":
    trainer()