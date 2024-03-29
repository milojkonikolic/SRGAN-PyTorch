import os
import argparse
import yaml
from tqdm import tqdm
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils.utils import get_logger, get_gpu, get_optimizer, save_weights, postprocess_image, ssim
from dataset.dataset import create_dataloader
from model.model import Generator, Discriminator
from model.loss import ContentLoss, AdversarialLoss


def train(config):
    
    logger = get_logger()
    train_dataloader = create_dataloader(config["Data"]["train_data_path"], config["Data"]["input_shape"], 
                                         config["Data"]["downsample_factor"], config["Data"]["gaussian_blur"], 
                                         config["Hyperparameters"]["batch_size"], config["Data"]["num_workers"], 
                                         shuffle=True)
    val_dataloader = create_dataloader(config["Data"]["val_data_path"], config["Data"]["input_shape"], 
                                       config["Data"]["downsample_factor"], config["Data"]["gaussian_blur"], 
                                       config["Hyperparameters"]["batch_size"], config["Data"]["num_workers"])
    gpu = get_gpu(config["Data"]["gpu"], logger)
    generator_net = Generator().train().cuda(gpu)
    discriminator_net = Discriminator(config["Data"]["input_shape"][0], config["Data"]["input_shape"][1]).train().cuda(gpu)
    generator_optimizer = get_optimizer(generator_net, config["Generator"]["optimizer"], config["Generator"]["lr"])
    generator_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=3)
    discriminator_optimizer = get_optimizer(discriminator_net, config["Discriminator"]["optimizer"], 
                                            config["Discriminator"]["lr"])
    discriminator_lr_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=3)
    content_loss = ContentLoss()
    adversarial_loss = AdversarialLoss()

    total_loss = 0
    global_step = 1
    weights_dir = os.path.join(config["Logging"]["train_logs"], "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        logger.info(f"Created dir for saving weights: {weights_dir}")
    tb_logs_dir = os.path.join(config["Logging"]["train_logs"], "tb_logs")
    if not os.path.exists(tb_logs_dir):
        os.mkdir(tb_logs_dir)
        logger.info(f"Created dir for saving tensorboard logs: {tb_logs_dir}")
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)

    for epoch in tqdm(range(config["Hyperparameters"]["epochs"])):
        pbar = tqdm(train_dataloader)
        for image_hr, image_lr in pbar:
            image_hr = image_hr.cuda(gpu)
            image_lr = image_lr.cuda(gpu)

            # Train discriminator less often than generator
            if global_step % config["Hyperparameters"]["generator_num"] == 0:
                discriminator_optimizer.zero_grad()
                # Generate gt for real image
                discriminator_real_gt = torch.ones((image_hr.shape[0], 1), device=gpu)
                # Get discriminator loss for high resolution image
                discriminator_pred_real = discriminator_net(image_hr)
                discriminator_loss_real = adversarial_loss(discriminator_pred_real, discriminator_real_gt)
                # Generate gt for generated image
                discriminator_gen_gt = torch.zeros((image_hr.shape[0], 1), device=gpu)
                # Get discriminator loss for generated image
                generator_image = generator_net(image_lr)
                discriminator_pred_gen = discriminator_net(generator_image)
                discriminator_loss_gen = adversarial_loss(discriminator_pred_gen, discriminator_gen_gt)
                discriminator_loss = discriminator_loss_real + discriminator_loss_gen
                discriminator_loss.backward()
                discriminator_optimizer.step()
                tb_writer.add_scalar("Discriminator Loss", discriminator_loss.item(), global_step)
                total_loss = discriminator_loss

            # Generator Learning
            generator_optimizer.zero_grad()
            generator_image = generator_net(image_lr)
            generator_loss = content_loss(image_hr, generator_image)
            generator_loss.backward()
            generator_optimizer.step()
            tb_writer.add_scalar("Generator Loss", generator_loss.item(), global_step)

            total_loss += generator_loss
            tb_writer.add_scalar("Total Loss", total_loss.item(), global_step)
            global_step += 1
            pbar.set_description(f"epoch: {epoch}/{config['Hyperparameters']['epochs']}, "
                                 f"loss: {round(float(total_loss), 5)}")

        generator_lr_scheduler.step(epoch)
        discriminator_lr_scheduler.step(epoch)
        tb_writer.add_scalar("Generator LR", generator_optimizer.param_groups[0]["lr"], global_step)
        tb_writer.add_scalar("Discriminator LR", discriminator_optimizer.param_groups[0]["lr"], global_step)

        save_weights(generator_net, weights_dir, "generator", epoch, logger)
        save_weights(discriminator_net, weights_dir, "discriminator", epoch, logger)

        # Calculate SSIM for valdation images
        print("Calculating SSIM on validation dataset")
        ssim_val = []
        num = 1
        for image_hr_val, image_lr_val in tqdm(val_dataloader):
            image_hr_val = image_hr_val.cuda(gpu)
            image_lr_val = image_lr_val.cuda(gpu)

            generator_image_val = generator_net(image_lr_val)
            generator_image_val_postp = postprocess_image(generator_image_val)
            image_hr_val_postp = postprocess_image(image_hr_val)
            ssim_idx_val = ssim(generator_image_val_postp, image_hr_val_postp)
            ssim_val.append(ssim_idx_val)

            # Add random val image to tensorboard
            if epoch % num == 0:
                tb_writer.add_image("generated_image", generator_image_val[0,:,:,:], global_step)
                tb_writer.add_image("low_res_image", image_lr_val[0,:,:,:], global_step)
                tb_writer.add_image("high_res_image", image_hr_val[0,:,:,:], global_step)
            num += 1
        
        ssim_mean_val = np.mean(np.array(ssim_val))
        tb_writer.add_scalar("Val SSIM", ssim_mean_val, global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Path to .yaml file with train configuration")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    train(config)
