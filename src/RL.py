import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_percls_celoss(logits, target):
    """
    Calculate the average loss for each class
    """
    loss = F.cross_entropy(logits, target, reduction='none')
    
    C = logits.size(1)  # 2
    
    mask_0 = (target == 0)
    mask_1 = (target == 1)
    
    # Calculate the average loss for class 0
    if mask_0.any():
        avg_loss_0 = loss[mask_0].mean()
    else:
        avg_loss_0 = torch.tensor(0.0, device=logits.device)
    
    # Calculate the average loss for class 1
    if mask_1.any():
        avg_loss_1 = loss[mask_1].mean()
    else:
        avg_loss_1 = torch.tensor(0.0, device=logits.device)
    
    per_class_avg_loss = torch.zeros((C), device=logits.device)
    per_class_avg_loss[0] = avg_loss_0
    per_class_avg_loss[1] = avg_loss_1
    
    return per_class_avg_loss

class ResidualLearningLoss(nn.Module):
    def __init__(self, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, target):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, target)

        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss

    def forward_percls(self, logit, target):
        # Cross entropy after balanced
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss_cls = compute_percls_celoss(logit_adjusted, target)

        # Regularization term
        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # [C, B]
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # [C, B]
        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # [C]

        estim_loss, _, _ = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)  # [C]
        estim_loss_cls = -estim_loss * self.cls_weight * 2  # [C]

        final_loss = ce_loss_cls + self.estim_loss_weight * estim_loss_cls

        return final_loss