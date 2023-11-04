import os
import json
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetBuilder(Dataset):
    def __init__(self, data_path, img_size, downsample_factor):
        self.data = self.read_data(data_path)
        self.img_size = img_size
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]
        image_hr = self.read_image(img_path)
        image_hr = self.preprocess_image(image_hr)
        # Create low resolution image using Gaussian filter and downsampling factor
        image_lr = self.read_image(img_path)
        image_lr = self.preprocess_image(image_lr, gaussian_blur=True, downsample=True)
        return image_hr, image_lr

    def read_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def read_image(self, img_path):
        if os.path.exists(img_path):
            image = cv.imread(img_path)
        else:
            raise f"Not found image {img_path}"
        return image

    def preprocess_image(self, image, gaussian_blur=False, downsample=False):
        if gaussian_blur:
            image = cv.GaussianBlur(image, (1, 1), cv.BORDER_DEFAULT)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        #image = image / 128. - 1.
        image = image / 256.
        org_height, org_width, _ = image.shape
        # Pad image if dimensions of the image are smaller than provided image size
        if org_height < self.img_size[1]:
            pad_val = [(int((self.img_size[1] - org_height) / 2.),
                        int((self.img_size[1] - org_height) / 2.)),
                       (int((self.img_size[0] - org_width) / 2.),
                        int((self.img_size[0] - org_width) / 2.)),
                       (0, 0)]
            image = np.pad(image, pad_val)
        # Downsample low resolution image
        if downsample:
            image = cv.resize(image, (self.img_size[0] // self.downsample_factor, 
                                      self.img_size[1] // self.downsample_factor))
        else:
            image = cv.resize(image, tuple(self.img_size))
        image = np.transpose(image, (2, 0, 1))
        return torch.FloatTensor(image)


def create_dataloader(data_path, img_size, downsample_factor, batch_size, num_workers=0, shuffle=False):
    dataset = DatasetBuilder(data_path, img_size, downsample_factor)
    dataloader = DataLoader(dataset=dataset, pin_memory=True, batch_size=batch_size, 
                                  num_workers=num_workers, shuffle=shuffle)
    return dataloader
