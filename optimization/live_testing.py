from torch import no_grad, max, device
from torch.jit import load
from torchvision import transforms
from cv2 import cvtColor, COLOR_BGR2RGB, VideoCapture, resize, imshow, waitKey, destroyAllWindows
from PIL.Image import fromarray

def predict_from_im(model, transform, image):
    # Convert OpenCV image to PIL Image
    color_converted_im = cvtColor(image, COLOR_BGR2RGB)
    pil_img = fromarray(color_converted_im)

    # Apply transforms to work with model
    img = transform['predict'](pil_img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with no_grad():
        outputs = model(img)
        _, preds = max(outputs, 1)
        print(f"Prediction: {preds}")
        
if __name__ == '__main__':
    # Set up GPU
    device = device("cpu")

    # Load model that we trained
    model = load("mobile_model_quantized.ptl", map_location=device('cpu')).to(device)
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

    # Set up the camera
    vc = VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        frame = resize(frame, (0,0), fx=0.5, fy=0.5)
        
        imshow("Detection", frame)
        predict_from_im(model, data_transforms, frame)

        if waitKey(1) == 27: # exit on ESC
            break

    vc.release()
    destroyAllWindows()
