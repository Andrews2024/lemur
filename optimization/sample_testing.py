import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

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

def predict_from_im(model, transform, image, class_names): 
    # Apply transforms to work with model
    pil_img = Image.open(image).convert('RGB')
    img = transform['predict'](pil_img)
    img = img.unsqueeze(0)
    img = img.to("cpu")

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        print(f"Prediction: {class_names[preds[0]]}")

if __name__ == '__main__':
    # Set up GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model that we trained
    model = torch.jit.load("full_model_quantized.pth").to("cpu")
    model.eval()

    # Use the same transforms as validation
    data_transforms = {
        'predict': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Test the model with images from the internet
    class_names = ['can', 'pasta']
    predict_from_im(model, data_transforms, "pasta-test.jpg", class_names)
    predict_from_im(model, data_transforms, "can-test.jpg", class_names)

# Note: if getting seg faults, check that device matches up