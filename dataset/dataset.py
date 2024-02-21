import os
import json
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetBuilder(Dataset):
    """
    This class builds a custom dataset for training a super-resolution model.
    Args:
        data_path (str): Path to the JSON file containing image paths.
        img_size (tuple): Size of the output images in the format (height, width).
        downsample_factor (int): Factor by which the low-resolution images are downsampled.
    Returns:
        torch.utils.data.Dataset: Custom dataset object.
    """
    def __init__(self, data_path, img_size, downsample_factor, gaussian_blur):
        """
        Initialize the DatasetBuilder.
        Args:
            data_path (str): Path to the JSON file containing image paths.
            img_size (tuple): Size of the output images in the format (height, width).
            downsample_factor (int): Factor by which the low-resolution images are downsampled.
            gaussian_blur (bool): If True apply Gaussian blur
        """
        self.data = self.read_data(data_path)
        self.img_size = img_size
        self.downsample_factor = downsample_factor
        self.gaussian_blur = gaussian_blur

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Get an item from the dataset.
        Args:
            item (int): Index of the item.
        Returns:
            torch.Tensor: High-resolution image tensor.
            torch.Tensor: Low-resolution image tensor.
        """
        img_path = self.data[item]
        image_hr = self.read_image(img_path)
        image_hr = self.preprocess_image(image_hr)
        # Create low resolution image using Gaussian filter and downsampling factor
        image_lr = self.read_image(img_path)
        image_lr = self.preprocess_image(image_lr, gaussian_blur=self.gaussian_blur, downsample=True)
        return image_hr, image_lr

    def read_data(self, data_path):
        """
        Read image paths from a JSON file.
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def read_image(self, img_path):
        """
        Read an image from the given path.
        """
        if os.path.exists(img_path):
            image = cv.imread(img_path)
        else:
            raise f"Not found image {img_path}"
        return image

    def preprocess_image(self, image, gaussian_blur=False, downsample=False):
        """
        Preprocess the image.
        Args:
            image (numpy.ndarray): Input image array.
            gaussian_blur (bool, optional): Whether to apply Gaussian blur. Default is False.
            downsample (bool, optional): Whether to downsample the image. Default is False.
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
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


def create_dataloader(data_path, img_size, downsample_factor, gaussian_blur, batch_size, num_workers=0, shuffle=False):
    """
    Create a data loader for the custom dataset.
    Args:
        data_path (str): Path to the JSON file containing image paths.
        img_size (tuple): Size of the output images in the format (height, width).
        downsample_factor (int): Factor by which the low-resolution images are downsampled.
        gaussian_blur (bool): If True apply Gaussian blur
        batch_size (int): Batch size for the data loader.
        num_workers (int, optional): Number of worker processes for data loading. Default is 0.
        shuffle (bool, optional): Whether to shuffle the data. Default is False.
    Returns:
        torch.utils.data.DataLoader: Data loader object.
    """
    dataset = DatasetBuilder(data_path, img_size, downsample_factor, gaussian_blur)
    dataloader = DataLoader(dataset=dataset, pin_memory=True, batch_size=batch_size, 
                                  num_workers=num_workers, shuffle=shuffle)
    return dataloader
