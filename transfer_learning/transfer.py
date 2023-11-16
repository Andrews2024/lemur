# Based off of code from here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
from torch import nn
from torchvision import models, transforms

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: Set up training data
# Set up transforms that crop images, normalize them, and performs some flipping for training images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets with labels

# Get pre-built model
mobilenet3 = models.mobilenet_v3_large()

# TODO: Prep model for training by freezing layers

# TODO: Train model on our data

# Save model
torch.save(mobilenet3.state_dict(), "mobilenet_model.pth")