from ultralytics import YOLO
# import os

from dataloading import create_dataloader

# Load a model
model = YOLO("models/arthropod_dectector_wave11_best.pt") # load .pt file

# Load images from a folder
folder_path = "examples/Entomoscope_examples/"
dataloader = create_dataloader(folder_path, crop_profile="Entomoscope")

results = []
for images, _ in dataloader:

    # Perform inference using YOLO model
    predictions = model.predict(images, show=False, save=True, save_txt=False)
    # predictions = model(images)
    results.append(predictions)


total_detections = 0

# Process the results
for result in results:
    # Process the predictions for each batch
    for prediction in result:
        total_detections += prediction.boxes.cls.cpu().numpy().shape[0]

print(f"Total number of detections: {total_detections}")