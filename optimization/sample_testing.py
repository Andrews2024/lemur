import torch
from torchvision import transforms

if __name__ == '__main__':
    # Set up GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model that we trained
    model = torch.load("full_model_quantized.pth").to(device)
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