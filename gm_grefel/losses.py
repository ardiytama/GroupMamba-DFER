import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleLoss(nn.Module):
    def __init__(self, lambda_anchor=10.0, lambda_center=10.0):
        super().__init__()
        self.lambda_anchor = lambda_anchor
        self.lambda_center = lambda_center

    def forward(self, outputs, target):
        log_probs, features, anchors = outputs
        anchors = anchors.float()
        loss_cls = F.nll_loss(log_probs.float(), target)
        # ||A A^T - I||_F^2
        eye = torch.eye(anchors.size(0), device=anchors.device)
        loss_anchor = ((anchors @ anchors.T - eye) ** 2).sum()
        # 0.5 * ||v_e - a_y||^2
        loss_center = 0.5 * ((features.float() - anchors[target]) ** 2).sum(dim=-1).mean()
        total = loss_cls + self.lambda_anchor * loss_anchor + self.lambda_center * loss_center
        return total, {'cls': loss_cls.item(), 'anchor': loss_anchor.item(), 'center': loss_center.item()}
