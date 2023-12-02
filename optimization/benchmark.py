from os import stat

if __name__ == '__main__':
    # Get size of all the files in MB
    full_original = round(stat('full_model_original.pth').st_size / 1e6, 2)
    mobile_original = round(stat('mobile_model_original.ptl').st_size / 1e6, 2)
    full_quantized = round(stat('full_model_quantized.pth').st_size / 1e6, 2)
    mobile_quantized = round(stat('mobile_model_quantized.ptl').st_size / 1e6, 2)

    print(f"Full original: {str(full_original)} MB\nMobile original: {str(mobile_original)} MB")
    print(f"Full quantized: {str(full_quantized)} MB\nMobile quantized: {str(mobile_quantized)} MB")

    # et time to load each model

    # Get time to get a prediction for each model
    
    # Get accuracy of each model on a test set of data