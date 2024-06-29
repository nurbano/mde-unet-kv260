import torch
from torchmetrics.functional.image import image_gradients
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn

class mde_loss(nn.Module):
    def __init__(self):
        super(mde_loss, self).__init__()

    def forward(self, y_true, y_pred, w1, w2, w3, w4, device):
        
        #L1 loss
        l_depth = torch.mean(torch.abs(y_pred - y_true), axis=-1)

        #L2 loss
        l2_depth = torch.mean(torch.pow(y_pred - y_true, 2), axis=-1)

        # edge loss for sharp edges
        dy_true, dx_true = image_gradients(y_true)
        dy_pred, dx_pred = image_gradients(y_pred)
        l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)
        
        # structural similarity loss
        ssim = StructuralSimilarityIndexMeasure(data_range=256,
                                        reduction='elementwise_mean'
                                      ,k1=0.01 ** 2
                                      , k2=0.03 ** 2
                                      , kernel_size=7,
                                        sigma=1.5).to(device)
        l_ssim = torch.clip((1 - ssim(y_pred, y_true)) * 0.5, 0, 1)
     
        return (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth)) + (w4 * torch.mean(l2_depth))