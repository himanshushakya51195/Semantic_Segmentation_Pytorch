import torch
import numpy as np
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils import base


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class IOU(base.Metric):
    __name__ = "IOU_Score"

    def __init__(self, activation=None, ignore_cls=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        if ignore_cls is None:
            self.ignore_cls = []
        else:
            self.ignore_cls = ignore_cls

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor):
        y_pr = self.activation(y_pr)
        assert y_pr.shape == y_gt.shape, f'y_pr and y_gt must have the same shape after {self.activation.__class__.__name__.lower()}'

        ious = np.array([], dtype=np.float64)

        for cls in y_gt.unique().cpu().numpy():
            if cls in self.ignore_cls:
                continue
            pr_inds = y_pr == cls
            gt_inds = y_gt == cls
            intersection = (pr_inds[gt_inds]).long().sum().data.cpu().item()
            union = pr_inds.long().sum().data.cpu().item() + gt_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                ious = np.append(ious, float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                iou = float(intersection) / float(max(union, 1))
                ious = np.append(ious, iou)

        return torch.tensor(ious, device=y_pr.device).mean()
