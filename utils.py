import os
import logging
import torch
import numpy as np
import cv2 as cv

def get_logger():
    """
    Get a logger instance for SRGAN.
    Returns:
        logging.Logger: Logger instance for SRGAN.
    """
    logger = logging.getLogger("SRGAN")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]-[%(filename)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def save_weights(model, weights_dir, model_name, epoch, logger):
    """
    Saves the state dictionary of a model to a file in the specified directory.
    Args:
        model (torch.nn.Module): Model to save weights for.
        weights_dir (str): Directory path to save the weights.
        model_name (str): Name of the model.
        epoch (int): Epoch number.
        logger (logging.Logger): Logger instance for logging.
    Returns:
        None
    """
    weights_path = os.path.join(weights_dir, f"srgan_{model_name}_ep{epoch}.pt")
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Model saved {weights_path}")


def get_gpu(gpu, logger=None):
    """
    Get the torch device for GPU or CPU.
    Args:
        gpu (str): Specify the device ("cpu" or "gpu").
        logger (logging.Logger, optional): Logger instance for logging. Default is None.
    Returns:
        torch.device: Torch device for computation.
    """
    if gpu == "cpu":
        if logger:
            logger.info("Selected CPU instead of GPU. Consider using GPU for better performance")
        return torch.device("cpu")
    else:
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid gpu {gpu} requested"
        if logger:
            logger.info(f"Selected gpu: GPU{gpu}")
        return torch.device(f"cuda:{gpu}")


def get_optimizer(model, opt, lr=0.001):
    """
    Get an optimizer for the model.
    Args:
        model (torch.nn.Module): Model for which optimizer is needed.
        opt (str): Name of the optimizer ("adam" or "sgd").
        lr (float, optional): Learning rate for the optimizer. Default is 0.001.
    Returns:
        torch.optim.Optimizer: Optimizer instance.
    """
    if opt.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                  f"For supported optimizers see documentation")
    return optimizer


def postprocess_image(image):
    """
    Postprocess the image tensor.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        numpy.ndarray: Postprocessed image array.
    """
    image = image[0,:,:,:].to("cpu").detach().numpy()
    image = np.moveaxis(image, 0, 2)
    image = image * 256
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image
