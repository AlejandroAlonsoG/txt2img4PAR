import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight

"""
This module implements the proposed augmented loss.

Usage:
    1. Place this file in the 'losses' folder of the original codebase.
    2. Set the 'positions_to_modify' variable to the indexes of your augmented attributes 
       in the original dataset.
    3. Ensure that these indexes do not correspond to any indirectly augmented attributes 
       in other versions of the datasets.
"""

@LOSSES.register("bceloss_augmented")
class BCELoss_augmented(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None, reduced_weight=0.5):
        super(BCELoss_augmented, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.reduced_weight = reduced_weight  # New parameter for reducing the weight of target 3

    def forward(self, logits, targets):
        logits = logits[0]

        # If the attribute is bald_head, short_skirt, or AgeLess16 we dont give it less weight 
        positions_to_modify =[0, 17, 34, 35] # AgeLess16 is on 35 for RAPzs
        for pos in positions_to_modify:
            if (targets[:, pos] == 3).any().item():
                mask = (targets[:, pos] == 3)
                targets[mask, pos] = 1

        valid_mask = targets != -1 # Mask for ignoring values where label = -1

        present_augmented_attributes = targets > 1 # Mask for all attributes that likelly have been generated

        modified_targets = torch.where(targets > 1, torch.ones_like(targets), targets) # Put 1's on the 3's for using them at first 
        
        loss_m = F.binary_cross_entropy_with_logits(logits, modified_targets.float(), reduction='none')

        loss_m = loss_m * valid_mask.float() # Remove from the loss all attributes that are -1 on the target vector

        # Check for NaNs
        if torch.isnan(logits).any() or torch.isnan(targets).any() or torch.isnan(modified_targets).any() or torch.isnan(loss_m).any():
            print("NaN detected during loss computation")
            exit(1)

        # Weight the loss for positions where the target was originally 3
        if self.reduced_weight < 1.0:
            loss_m = loss_m * torch.where(present_augmented_attributes, torch.tensor(self.reduced_weight, dtype=loss_m.dtype, device=loss_m.device), torch.tensor(1.0, dtype=loss_m.dtype, device=loss_m.device))

        if self.sample_weight is not None:
            targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
            sample_weight = ratio2weight(targets_mask, self.sample_weight)
            loss_m = loss_m * sample_weight.cuda()

        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        if (loss_m < 0).any():
            print("Negative values detected in loss_m")
            exit(1)

        return [loss], [loss_m]