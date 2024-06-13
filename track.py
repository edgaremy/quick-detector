import cv2
from ultralytics import YOLO
# import os

from dataloading import create_dataloader


BATCH_SIZE = 1 # WARNING: The batch currently give batch_size times the SAME picture
SHOW_TRACKING = True

DISPLAY_SIZE = (1000, 1000)

# Load images from a folder
folder_path = "./examples/Entomoscope_sequence/"
dataloader = create_dataloader(folder_path, crop_profile="Entomoscope", batch_size=BATCH_SIZE, use_cv2=True)

# Load the YOLOv8 model
model = YOLO("models/arthropod_dectector_wave11_best.pt") # load .pt file

trackings = []
for image, _, image_dir in dataloader:
    # # Load the image
    image = cv2.imread(image_dir[0])

    # Run YOLOv8 tracking on the image, persisting tracks between frames
    results = model.track(image, persist=True, tracker="entomoscope_tracking.yaml")
    print(results[0].boxes)
    # trackings.append(results[0].boxes.id.numpy().tolist())
    
    if SHOW_TRACKING:
        # Visualize the results on the image
        annotated_image = results[0].plot()

        # Display the annotated image
        cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_image, DISPLAY_SIZE))
        cv2.waitKey(0)

cv2.destroyAllWindows()

print(trackings)
# Count the total number of detections
total_detections = 0
# Process the results
for result in trackings:
    # Create a set to store the distinct indexes found
    distinct_indexes = set()
    # Process the predictions for each batch
    for prediction in result:
        # Get the indexes of the detected objects
        # indexes = prediction.boxes.id.cpu().numpy()
        print(prediction.boxes.id)
        #convert tensor list to list
        # indexes = prediction.boxes.id.numpy().tolist()
        # Add the distinct indexes to the set
        # distinct_indexes.update(indexes)
    # Increment the total detections by the number of distinct indexes found
    total_detections += len(distinct_indexes)

print(f"Total number of detections: {total_detections}")