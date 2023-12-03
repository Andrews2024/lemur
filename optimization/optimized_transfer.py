# Based off of code from here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html, https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html, 
import torch
from torch import nn
from torchvision import models, transforms, datasets
import torchvision.models.quantization as qmodels
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.quantization import convert
from torch.nn.utils import prune

from time import time
import copy
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
  """
  Support function for model training.

  Args:
    model: Model to be trained
    criterion: Optimization criterion (loss)
    optimizer: Optimizer to use for training
    scheduler: Instance of ``torch.optim.lr_scheduler``
    num_epochs: Number of epochs
    device: Device to run the training on. Must be 'cpu' or 'cuda'
  """
  since = time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'validate']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'validate' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

def create_combined_model(model_fe):
  # Step 1. Isolate the feature extractor.
  model_fe_features = nn.Sequential(
    model_fe.quant,  # Quantize the input
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    model_fe.layer4,
    model_fe.avgpool,
    model_fe.dequant,  # Dequantize the output
  )

  # Step 2. Create a new "head"
  new_head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2),
  )

  # Step 3. Combine, and don't forget the quant stubs.
  new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
  )
  return new_model

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path).convert('RGB')
    img = data_transforms['validate'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)
        plt.show()

if __name__ == '__main__':
    # Set device to GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    optimizations = ["Quantization", 'Pruning']
    optimization = optimizations[1] # Change which training is done based on whether we're doing quantization

    # Get pre-built model
    if optimization == "Quantization":
      model_conv = qmodels.resnet18(weights='DEFAULT', progress=True, quantize=True)
    
    else:
      model_conv = models.resnet18(weights='IMAGENET1K_V1') # For both pruning and unoptimized

    # Prep model for training by freezing layers
    for param in model_conv.parameters():
      param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    if optimization == "Quantization":
      new_model = create_combined_model(model_conv)
      new_model = new_model.to('cpu')

      criterion = nn.CrossEntropyLoss()

      # Note that we are only training the head.
      optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

      # Decay LR by a factor of 0.1 every 7 epochs
      exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

      # Train the model
      new_model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device='cpu')

      # Convert the model to a quantized model
      new_model = convert(new_model, inplace=False)
    
    else:
      new_model = model_conv.to(device) # Put the model on CPU/ GPU for training

      criterion = nn.CrossEntropyLoss() # Loss function for training

      # Observe that only parameters of final layer are being optimized as opposed to before.
      optimizer_conv = optim.SGD(new_model.fc.parameters(), lr=0.001, momentum=0.9)

      # Decay LR by a factor of 0.1 every 7 epochs
      exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

      # Train the model
      new_model = train_model(new_model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25, device=device)

    # Test the model with images from the internet
    visualize_model_predictions(new_model, "pasta-test.jpg")
    visualize_model_predictions(new_model, "can-test.jpg")

    if optimization == "Quantization":
        # Save model for Raspberry Pi
        scripted_model = torch.jit.script(new_model)
        mobile_model = optimize_for_mobile(scripted_model)
        mobile_model._save_for_lite_interpreter("mobile_model_quantized.ptl")

        # Save model for computer
        torch.jit.save(scripted_model, "full_model_quantized.pth")
    
    else:
      # Save model for Raspberry Pi
      scripted_model = torch.jit.script(new_model)
      mobile_model = optimize_for_mobile(scripted_model)
      mobile_model._save_for_lite_interpreter("mobile_model_original.ptl")

      # Save model for computer
      torch.save(new_model, "full_model_original.pth")

      if optimization == "Pruning":
        parameters_to_prune = ( # Pick some modules to prune
            (new_model.layer2[1].conv1, "weight"),
            (new_model.layer2[1].conv2, "weight"),
            (new_model.layer4[1].conv1, "weight"),
            (new_model.layer4[1].conv2, "weight"),
          )

        # Apply global unstructured pruning to selected layers
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=.2) 
        
        for module in parameters_to_prune:
          prune.remove(module[0], module[1]) # Makes the pruning permanent

        # Test the model with images from the internet
        visualize_model_predictions(new_model, "pasta-test.jpg")
        visualize_model_predictions(new_model, "can-test.jpg")
        
        # Save model for Raspberry Pi
        scripted_model_pruned = torch.jit.script(new_model)
        mobile_model_pruned = optimize_for_mobile(scripted_model_pruned)
        mobile_model_pruned._save_for_lite_interpreter("mobile_model_pruned.ptl")

        # Save model for computer
        torch.save(new_model, "full_model_pruned.pth")
    
# for layer_name, param in new_model.named_parameters(): # Look at different modules here
#   print(f"layer name: {layer_name} has {param.shape}")