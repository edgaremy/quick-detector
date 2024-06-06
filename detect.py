from ultralytics import YOLO
import os

from dataloading import create_dataloader

# Load a model
model = YOLO("models/arthropod_dectector_wave10_best.pt") # load .pt file

# Load images from a folder
folder_path = "examples/"
dataloader = create_dataloader(folder_path)

results = []
for images, _ in dataloader:

    model.predict(images, show=False, save=True, save_txt=False)

#     # Perform inference using YOLO model
#     predictions = model(images)
#     results.append(predictions)

# # Process the results
# for result in results:
#     # Process the predictions for each batch
#     for prediction in result:
#         # Process individual prediction
#         # TODO: Add your code here to handle the predictions
#         pass