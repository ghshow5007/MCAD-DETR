import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideVariFocalLoss(nn.Module):
    """SlideVariFocalLoss (SVFL)."""

    def __init__(self):
        super(SlideVariFocalLoss, self).__init__()

    def forward(self, pred, true, one_hot, auto_iou=0.5):
        """Compute SVFL with sliding modulation based on auto_iou."""
        loss = self.loss_fcn(pred, true, one_hot)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        return loss.mean(1).sum()

    def loss_fcn(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Varifocal-style classification loss core used by SVFL."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight)
        return loss


def Focaler_MPDIoU(box1, box2, xywh=True, mpdiou_hw=1, eps=1e-7, d=0.0, u=0.95):
    """Focaler-MPDIoU.

    This is the Focaler-MPDIoU variant of IoU, combining Focaler-IoU
    re-scaling with the MPDIoU corner-distance penalty.

    Args:
        box1 (torch.Tensor): shape (N, 4) or (1, 4), boxes in xywh or xyxy format.
        box2 (torch.Tensor): shape (M, 4), boxes in xywh or xyxy format.
        xywh (bool): If True, inputs are in (x, y, w, h) format; otherwise (x1, y1, x2, y2).
        mpdiou_hw (float): MPDIoU normalization factor for the corner distance penalty.
        eps (float): Small epsilon for numerical stability.
        d (float): Lower clipping bound for Focaler-IoU.
        u (float): Upper clipping bound for Focaler-IoU.

    Returns:
        torch.Tensor: Focaler-MPDIoU value(s).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    # Focaler-IoU re-scaling
    iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00, u=0.95
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # Focaler-MPDIoU


class GOIULoss(nn.Module):
    """GIoU-like loss based on Focaler_MPDIoU.

    This loss is implemented as ``1 - Focaler_MPDIoU`` and behaves as a
    GIoU-style bounding box regression loss, supporting ``mean`` / ``sum`` /
    ``none`` reductions.
    """


    def __init__(self, xywh=True, mpdiou_hw=1, d=0.0, u=0.95, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.xywh = xywh
        self.mpdiou_hw = mpdiou_hw
        self.d = d
        self.u = u
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute GOIU loss between predicted and target boxes."""
        iou = Focaler_MPDIoU(
            pred_boxes,
            target_boxes,
            xywh=self.xywh,
            mpdiou_hw=self.mpdiou_hw,
            d=self.d,
            u=self.u,
        )
        loss = 1.0 - iou
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
