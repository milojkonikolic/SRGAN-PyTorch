import torch.nn as nn


class ContentLoss(nn.modules.Module):
    """
    Content loss module.
    This module calculates the Mean Squared Error (MSE) loss between the ground truth image 
    and the generated image.
    Args:
        None
    Returns:
        torch.Tensor: Content loss value.
    """
    # TODO: Change the loss function
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, gt_image, gen_image):
        """
        Forward pass through the content loss module.
        Args:
            gt_image (torch.Tensor): Ground truth image tensor.
            gen_image (torch.Tensor): Generated image tensor.
        Returns:
            torch.Tensor: Content loss value.
        """
        return self.mse_loss(gt_image, gen_image)

class AdversarialLoss(nn.modules.Module):
    """
    Adversarial loss module.
    This module calculates the Binary Cross Entropy (BCE) loss between the predicted 
    probabilities and the ground truth labels.
    Args:
        None
    Returns:
        torch.Tensor: Adversarial loss value.
    """
    # TODO: Change the loss function
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()
    
    def forward(self, pred, gt):
        """
        Forward pass through the adversarial loss module.
        Args:
            pred (torch.Tensor): Predicted probabilities tensor.
            gt (torch.Tensor): Ground truth labels tensor.
        Returns:
            torch.Tensor: Adversarial loss value.
        """
        return self.adversarial_loss(pred, gt)

