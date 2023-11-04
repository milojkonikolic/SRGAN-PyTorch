import torch
import torch.nn as nn


class ContentLoss(nn.modules.Module):
    # TODO: Change the loss function
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, gt_image, gen_image):
        return self.mse_loss(gt_image, gen_image)

class AdversarialLoss(nn.modules.Module):
    # TODO: Change the loss function
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()
    
    def forward(self, pred, gt):
        return self.adversarial_loss(pred, gt)

