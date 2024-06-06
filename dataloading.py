from ultralytics import YOLO
import torch
import torchvision
from torchvision.transforms import v2

# # Load a model
# model = YOLO("models/arthropod_dectector_wave10_best.pt") # load .pt file

# # folder_path = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG"
# folder_path = "examples/"
def create_dataloader(folder_path):

    def horizontal_crop(image, left_limit=415, right_limit=2380):
        w, h = image.size
        new_w = right_limit - left_limit
        return v2.functional.crop(image, top=0, left=left_limit, height=h, width=new_w)

    transform = v2.Compose([
        v2.Lambda(horizontal_crop),
        v2.Resize((640, 640)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # Create a Torch DataLoader from image directory
    dataset = torchvision.datasets.ImageFolder(folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader