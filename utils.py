import os
import logging
import torch


def get_logger():
    logger = logging.getLogger("SRGAN")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]-[%(filename)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def save_weights(model, weights_dir, model_name, epoch, logger):
    weights_path = os.path.join(weights_dir, f"srgan_{model_name}_ep{epoch}.pt")
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Model saved {weights_path}")


def get_gpu(gpu, logger):
    if gpu == "cpu":
        logger.info("Selected CPU instead of GPU. Consider using GPU for better performance")
        return torch.device("cpu")
    else:
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid gpu {gpu} requested"
        logger.info(f"Selected gpu: GPU{gpu}")
        return torch.device(f"cuda:{gpu}")


def get_optimizer(model, opt, lr=0.001):
    if opt.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                  f"For supported optimizers see documentation")
    return optimizer
