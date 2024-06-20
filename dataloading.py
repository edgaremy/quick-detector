import os
import cv2
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import v2

import numpy as np

class NoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.img_dir = os.listdir(dataset_path)
        # Eclude non png and jpg/jpeg files
        self.img_dir = [f for f in self.img_dir if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
        self.img_dir.sort()
        for i in range(len(self.img_dir)):
            self.img_dir[i] = os.path.join(dataset_path, self.img_dir[i])
        self.transform = transform


    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0, img_path # label is always 0

    def __len__(self):
        return len(self.img_dir)

class NoLabelDatasetCV2(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.img_dir = os.listdir(dataset_path)
        # Eclude non png and jpg/jpeg files
        self.img_dir = [f for f in self.img_dir if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
        self.img_dir.sort()
        for i in range(len(self.img_dir)):
            self.img_dir[i] = os.path.join(dataset_path, self.img_dir[i])
        self.transform = transform


    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
    
        # WARNING: fixed naming means all images in
        # a batch will actually look like the first one :
        image_save_dir = "temp.jpg"
        
        save_image(image, image_save_dir)
        # cv2_image = cv2.imread("temp.jpg")
        # print(image.shape)
        # print(cv2_image.shape)
        # remove temp image
        # os.remove("temp.jpg")
        return image, 0, image_save_dir # label is always 0

    def __len__(self):
        return len(self.img_dir)

def create_dataloader(folder_path, crop_profile="None", batch_size=32, use_cv2=False):

    # horizontal crop for the Entomoscope images crop profile
    def horizontal_crop(image, left_limit=415, right_limit=2380):
        w, h = image.size(2), image.size(1)
        new_w = right_limit - left_limit
        return v2.functional.crop(image, top=0, left=left_limit, height=h, width=new_w)

    if crop_profile == "Entomoscope":
        transform = v2.Compose([
            v2.Lambda(horizontal_crop),
            v2.Resize((640, 640)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    else: # Default loading profile
        transform = v2.Compose([
            v2.Resize((640, 640)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    # Create a Torch DataLoader from image directory
    # dataset = torchvision.datasets.ImageFolder(folder_path, transform=transform)
    if use_cv2:
        dataset = NoLabelDatasetCV2(folder_path, transform=transform)
    else:
        dataset = NoLabelDataset(folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

