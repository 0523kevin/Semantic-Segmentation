import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def init_loss(name):
    loss_dict = {
        'CE' : nn.CrossEntropyLoss(),
        'BCE' : nn.BCEWithLogitsLoss(),
        'softCE' : smp.losses.SoftCrossEntropyLoss(reduction='mean', smooth_factor=0.1),
        'softBCE' : smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1),
        'dice' : smp.losses.DiceLoss(mode='multilabel', smooth=0.1), #DiceLoss(), 
        'dicefocal' : DiceFocalLoss(0.9, 0.1),
        'tversky' : smp.losses.TverskyLoss(mode='multilabel', smooth=0.1)
        
    }
    return loss_dict[name]




from numpy.lib.arraysetops import union1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class custom_CrossEntropyLoss(nn.Module):

    def __init__(self, weights):
        nn.Module.__init__(self)
        self.CEL = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())


    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.CEL(pred[i], target) * weights[i]
            return loss

        else:
            return self.CEL(pred, target)
          
          
class mIoULoss(nn.Module):
    """
    code reference: https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
    """
    def __init__(self, weight=None, size_average=True, n_classes=11):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, pred, target, smooth = 1e-6):
        """
        pred: y_pred (N,C,H,W)
        target: y_true (should be scattered into N,C,H,W shaped tensor)
        """

        N = pred.size()[0]

        pred = F.softmax(pred, dim = 1)
        target_one_hot = self._to_one_hot(target)

        # intersection (numerator)
        intersec = pred * target_one_hot
        intersec = intersec.view(N, self.classes, -1).sum(2)  # sum over all pixels NxCxHxW => NxC

        # union (denominator)
        union = pred + target_one_hot - (pred*target_one_hot)
        union = union.view(N,self.classes,-1).sum(2)

        loss = (intersec+smooth)/(union+smooth)

        return -loss.mean() # miou는 최대화 문제이므로 최소화로 문제를 바꿔서 생각해줘야.

    
    def _to_one_hot(self, target):
        n,h,w = target.size()
        one_hot = torch.zeros(n,self.classes,h,w).cuda().scatter_(1, target.view(n,1,h,w), 1)
        return one_hot


class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()
        self.DL = smp.losses.DiceLoss(mode = "multilabel")

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.DL(pred[i], target) * weights[i]
            return loss

        else:
            return self.DL(pred, target)


class DiceCELoss(nn.Module):

    def __init__(self, dice_weight, ce_weight):

        super().__init__()
        self.DL = DiceLoss()
        self.CEL = custom_CrossEntropyLoss()
        self.weights = [dice_weight, ce_weight]

    def forward(self, pred, target):
        return self.weights[0] * self.DL(pred, target) + self.weights[1] * self.CEL(pred, target)


class FocalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.FL = smp.losses.FocalLoss(mode = "multilabel")

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.FL(pred[i], target) * weights[i]
            return loss

        else:
            return self.FL(pred, target)


class DiceFocalLoss(nn.Module):

    def __init__(self, dice_weight, focal_weight):

        super(DiceFocalLoss, self).__init__()
        self.DL = DiceLoss()
        self.FL = FocalLoss()
        self.weights = [dice_weight, focal_weight]

    def forward(self, pred, target):
        return self.weights[0]*self.DL(pred, target) + self.weights[1] * self.FL(pred, target)
    



# import warnings
# from typing import Callable, List, Optional, Sequence, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.loss import _Loss

# from monai.losses import DiceLoss, FocalLoss
# from monai.networks import one_hot
# from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after


# ## Modified to correct include_background error for DiceFocalLoss
# class DiceFocalLoss(_Loss):
#     """
#     Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
#     The details of Dice loss is shown in ``monai.losses.DiceLoss``.
#     The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

#     """

#     def __init__(
#         self,
#         include_background: bool = True,
#         to_onehot_y: bool = False,
#         sigmoid: bool = False,
#         softmax: bool = False,
#         other_act: Optional[Callable] = None,
#         squared_pred: bool = False,
#         jaccard: bool = False,
#         reduction: str = "mean",
#         smooth_nr: float = 1e-5,
#         smooth_dr: float = 1e-5,
#         batch: bool = False,
#         gamma: float = 2.0,
#         focal_weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
#         lambda_dice: float = 1.0,
#         lambda_focal: float = 1.0,
#     ) -> None:
#         super().__init__()
#         self.dice = DiceLoss(
#             include_background=include_background,
#             sigmoid=sigmoid,
#             softmax=softmax,
#             other_act=other_act,
#             squared_pred=squared_pred,
#             jaccard=jaccard,
#             reduction=reduction,
#             smooth_nr=smooth_nr,
#             smooth_dr=smooth_dr,
#             batch=batch,
#         )
#         self.focal = FocalLoss(gamma=gamma, weight=focal_weight, reduction=reduction)
#         if lambda_dice < 0.0:
#             raise ValueError("lambda_dice should be no less than 0.0.")
#         if lambda_focal < 0.0:
#             raise ValueError("lambda_focal should be no less than 0.0.")
#         self.lambda_dice = lambda_dice
#         self.lambda_focal = lambda_focal
#         self.to_onehot_y = to_onehot_y
#         self.include_background = include_background


#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input: the shape should be BNH[WD]. The input should be the original logits
#                 due to the restriction of ``monai.losses.FocalLoss``.
#             target: the shape should be BNH[WD] or B1H[WD].

#         Raises:
#             ValueError: When number of dimensions for input and target are different.
#             ValueError: When number of channels for target is neither 1 nor the same as input.

#         """
#         if len(input.shape) != len(target.shape):
#             raise ValueError("the number of dimensions for input and target should be the same.")

#         n_pred_ch = input.shape[1]

#         if self.to_onehot_y:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
#             else:
#                 target = one_hot(target, num_classes=n_pred_ch)

#         dice_loss = self.dice(input, target)
                
#         if not self.include_background:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `include_background=False` ignored.")
#             else:
#                 # if skipping background, removing first channel
#                 target = target[:, 1:]
#                 input = input[:, 1:]

#         focal_loss = self.focal(input, target)
#         total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
#         return total_loss
