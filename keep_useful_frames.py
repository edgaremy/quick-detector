from ultralytics import YOLO
import os
import shutil

from dataloading import create_dataloader

output_folder = "./output"
# make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load a model
model = YOLO("models/arthropod_dectector_wave16_best.pt") # load .pt file

# Load images from a folder
folder_path = "examples/Entomoscope_examples/"
dataloader = create_dataloader(folder_path, crop_profile="Entomoscope")

results = []
paths = []
for images, _, img_paths in dataloader:

    # Perform inference using YOLO model
    predictions = model.predict(images, show=False, save=True, save_txt=False)
    # predictions = model(images)
    results.append(predictions)
    paths.append(img_paths)




number_of_kept_frames = 0

# If something is detected, copy the image to the output folder
for b, result in enumerate(results): # for each batch
    for prediction, img_path in zip(result, paths[b]): # for each image in the batch
        if prediction.boxes.cls.cpu().numpy().shape[0] > 0: # if somthing is detected
            # copy the image to the output folder
            shutil.copy(img_path, output_folder)
            number_of_kept_frames += 1
            

print(f"Number of frames with detections: {number_of_kept_frames}")
# total_detections = 0

# # Process the results
# for result in results:
#     # Process the predictions for each batch
#     for prediction in result:
#         total_detections += prediction.boxes.cls.cpu().numpy().shape[0]

# print(f"Total number of detections: {total_detections}")