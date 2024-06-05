from ultralytics import YOLO
import os

# Load a model
model = YOLO("models/arthropod_dectector_wave10_best.pt") # load .pt file

# folder_path = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG"
folder_path = "examples/"

for folder_name in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, folder_name)

    file_paths = []
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

        extensions = set()
        for file_path in file_paths:
            file_name, file_extension = os.path.splitext(file_path)
            extensions.add(file_extension)

        print("List of file extensions:")
        print(list(set(extensions)))


    chunk_size = 100
    num_chunks = len(file_paths) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        model.predict(file_paths[start:end], show=False, save=True, save_txt=False)

    remaining_files = len(file_paths) % chunk_size
    if remaining_files > 0:
        start = num_chunks * chunk_size
        end = start + remaining_files
        model.predict(file_paths[start:end], show=False, save=True, save_txt=False)