from ultralytics import YOLO
# import os

from dataloading import create_dataloader

### PARAMETERS ###
# Path to the YOLO model
MODEL_PATH = "models/arthropod_dectector_wave16_best.pt"  # Replace with your model path
FOLDER_PATH = "examples/Entomoscope_examples/"  # Replace with your input folder path
OUTPUT_PATH = "./runs/detect"  # Replace with your desired output folder path

CROP_PROFILE = "Entomoscope"  # Set to "None" for other image sources
##################

# Load model
model = YOLO(MODEL_PATH) # load .pt file

# Load images from input folder
folder_path = "examples/Entomoscope_examples/"
dataloader = create_dataloader(FOLDER_PATH, crop_profile=CROP_PROFILE)

results = []
for images, _, _ in dataloader:

    # Perform inference using YOLO model
    predictions = model.predict(images, show=False, save=True, project="./runs/detect", save_txt=False)
    # predictions = model(images)
    results.append(predictions)


total_detections = 0

# Process the results
for result in results:
    # Process the predictions for each batch
    for prediction in result:
        print(f"Number of detections: {prediction.boxes.cls.cpu().numpy().shape[0]}")
        total_detections += prediction.boxes.cls.cpu().numpy().shape[0]

print(f"Total number of detections: {total_detections}")