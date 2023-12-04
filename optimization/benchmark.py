from os import stat
from time import time
from random import shuffle
import torch

def load_time(filename: str) -> float:
    '''Get wall time to load the provided model'''
    file_type = filename.split('.')[1]
    device = torch.device("cpu")

    if file_type == "ptl" or "quantized" in filename:
        start = time()
        model = torch.jit.load(filename).to(device)
        end = time()
        
        model.eval()
    
    else:
        start = time()
        model = torch.load(filename).to(device)
        end = time()
        
        model.eval()

    return end - start

if __name__ == '__main__':
    # All current models
    model_files = ['full_model_original.pth', 'mobile_model_original.ptl', 'full_model_quantized.pth',
                   'mobile_model_quantized.ptl', 'full_model_pruned.pth', 'mobile_model_pruned.ptl']
    shuffle(model_files) # Shuffle the list so when we run multiple tests we don't have the same starting file

    # Get time to load each model
    for model in model_files:
        print(model)
        file_size = round(stat(model).st_size / 1e6, 2)
        model_load_time = load_time(model)

        print(f"Model: {model.split('.')[0]}, Size: {file_size} MB, Load time: {model_load_time}\n")
    
    # Get time to get a prediction for each model
    
    # Get accuracy of each model on a test set of data