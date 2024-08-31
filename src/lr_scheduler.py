import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import math


class CustomLRScheduler(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_lr_start=0.0,
        warmup_total_iters=100,
        total_epochs=800,
        T=200,
        no_aug_iter=200,
        min_lr=0.2,
    ):
        self.warmup_lr_start = warmup_lr_start
        self.warmup_total_iters = warmup_total_iters
        self.no_aug_iter = no_aug_iter
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.T = T
        super(CustomLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_total_iters:
            lrs = [
                (lr - self.warmup_lr_start)
                * pow(self.last_epoch / float(self.warmup_total_iters), 2)
                + self.warmup_lr_start
                for lr in self.base_lrs
            ]
            return lrs
        elif self.last_epoch > (self.total_epochs - self.no_aug_iter):
            return [self.min_lr for _ in self.base_lrs]
        else:
            cur_epoch = self.last_epoch - self.warmup_total_iters
            T_n = (cur_epoch / self.T) * 2 * math.pi
            vf = (math.cos(T_n) + 1) / 2
            lrs = [vf * (lr - self.min_lr) +
                   self.min_lr for lr in self.base_lrs]
            return lrs
