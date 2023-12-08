from time import time
import torch
from torchvision import transforms
from PIL import Image
import cv2 as cv

def predict_from_im(model, transform, image):
    # Convert OpenCV image to PIL Image
    color_converted_im = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted_im)

    # Apply transforms to work with model
    img = transform['predict'](pil_img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        #print(f"Prediction: {preds[0]}")

def get_prediction_timed():
    pass

def get_prediction_fps(model, data_transforms):
    # Set up the camera
    vc = cv.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    # Start frame counting
    frame_count = 0
    prev_time = time()

    while rval:
        rval, frame = vc.read()
        frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
        
        cv.imshow("Detection", frame)
        predict_from_im(model, data_transforms, frame)

        frame_count += 1
        curr_time = time()

        if curr_time - prev_time > 1: # If more than one second has passed
            print(f"{frame_count / (curr_time - prev_time)} FPS")
            
            # Reset timing and frame counting
            prev_time = curr_time
            frame_count = 0

        if cv.waitKey(1) == 27: # exit on ESC
            break

    vc.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    # Use models from optimization module
    model_files = ['optimization/mobile_model_original.ptl', 'optimization/mobile_model_quantized.ptl']

    torch.backends.quantized.engine = 'qnnpack'
    
    # Set up CPU
    device = torch.device("cpu")

    # Load model that we trained
    model = torch.jit.load(model_files[0], map_location=torch.device('cpu')).to(device)
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
    
    # Get FPS for each model
    get_prediction_fps(model, data_transforms)

    # Get counts of each prediction (e.g. # pred = 0 vs. # pred = 1)