import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch import nn

from utils import get_logger, get_device, get_optimizer
from dataset import create_dataloader
from model import Generator, Discriminator
from loss import ContentLoss, AdversarialLoss


def train(config):
    
    logger = get_logger()
    train_dataloader = create_dataloader(config["data_path"], config["input_shape"], config["downsaple_factor"],
                                         config["batch_size"], config["num_workers"])

    generator_net = Generator()
    discriminator_net = Discriminator(config["input_shape"][0], config["input_shape"][1])
    device = get_device(config["device"])
    generator_optimizer = get_optimizer(config["optimizer"], config["lr"])
    discriminator_optimizer = get_optimizer(config["optimizer"], config["lr"])
    content_loss = ContentLoss()
    adversarial_loss = AdversarialLoss()

    weights_dir = os.path.join(config["train_logs"], "weights")
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
        logger.info(f"Created dir for saving weights: {weights_dir}")
    tb_logs_dir = os.path.join(config["train_logs"], "tb_logs")
    if not os.path.exists(tb_logs_dir):
        os.mkdir(tb_logs_dir)
        logger.info(f"Created dir for saving tensorboard logs: {tb_logs_dir}")

    for epoch in tqdm(range(config["epochs"])):
        pbar = tqdm(train_dataloader)
        for image_hr, image_lr in pbar:
            pass
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser():
    parser.add_argument("--config", type=str, default='',
                        help="Path to .yaml file with train configuration")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    train(config)
