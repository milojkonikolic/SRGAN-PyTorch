import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2 as cv

from model import Generator
from utils import postprocess_image


def read_image(img_path):
    """
    Read and preprocess an image from the given path.
    Args:
        img_path (str): Path to the image file.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = cv.imread(img_path)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = image / 256.
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return torch.FloatTensor(image).cuda()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-model-path", type=str, default='',
                        help="Path to trained generator model")
    parser.add_argument("--input-dir", type=str, default='',
                        help="Path to directory with images for test")
    parser.add_argument("--out-dir", type=str, default='',
                        help="Path to output directory for saving images")
    args = parser.parse_args()

    # Load generator model
    generator_model = Generator()
    generator_model.load_state_dict(torch.load(args.gen_model_path))
    generator_model.eval().cuda()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # Generate image for each low resoltion image in the input-dir
    for img_file in tqdm(os.listdir(args.input_dir)):
        img_path = os.path.join(args.input_dir, img_file)
        image = read_image(img_path)
        gen_image = generator_model(image)
        gen_image = postprocess_image(gen_image)
        out_img_path = os.path.join(args.out_dir, img_file)
        cv.imwrite(out_img_path, gen_image)
