# Based off of code from here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
from torch import nn
from torchvision import models, transforms, datasets

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
    'validate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets with labels
image_datasets = {x: datasets.ImageFolder(x, data_transforms[x]) for x in ['train', 'validate']} # Training and validation for all items
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'validate']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}
class_names = image_datasets['train'].classes
print(class_names)

# Get pre-built model
mobilenet3 = models.mobilenet_v3_large()

# TODO: Prep model for training by freezing layers

# TODO: Train model on our data

# Save model
torch.save(mobilenet3.state_dict(), "mobilenet_model.pth")