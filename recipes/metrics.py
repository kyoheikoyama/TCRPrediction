
from dataclasses import dataclass, asdict
from torch import nn
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.special import softmax


@dataclass
class ROC_AUC(nn.Module):
    def __init__(self):
        super().__init__()

    def _roc_auc_compute_fn(self, hh, yy):
        yy = torch.clone(yy)
        yy = yy.cpu().detach().numpy()
        hh = hh.cpu().detach().numpy()
        hh = softmax(hh, axis=1)[:,1]
        try:
            return torch.tensor(roc_auc_score(yy, hh))
        except:
            return torch.tensor(0.5)

        

    def forward(self, hh, yy) -> torch.tensor:
        return self._roc_auc_compute_fn(hh, yy).float()


class PR_AUC(nn.Module):
    def __init__(self, class_to_predict):
        super().__init__()
        self.class_to_predict = class_to_predict

    def _pr_auc_compute_fn(self, hh, yy):
        yy = torch.clone(yy)
        yy = yy.cpu().detach().numpy()
        hh = hh.cpu().detach().numpy()
        hh = softmax(hh, axis=1)[:, self.class_to_predict]
        precision, recall, thresholds = precision_recall_curve(yy, hh, pos_label=self.class_to_predict)
        args = np.argsort(recall)
        precision, recall = precision[args], recall[args]
        return torch.tensor(auc(recall, precision))

    def forward(self, hh, yy) -> torch.tensor:
        return self._pr_auc_compute_fn(hh, yy).float()



