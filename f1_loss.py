
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class F1Loss(nn.Module):
    """ F1Loss
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        f1_loss = 2*r*p / (p + r)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(F1Loss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, input, target):
        N, C = input.size()[:2]
        _, predict = torch.max(input, 1)# # (N, C, *) ==> (N, 1, *)

        predict = predict.view(N, 1, -1) # (N, 1, *)
        target = target.view(N, 1, -1) # (N, 1, *)
        last_size = target.size(-1)

        ## convert predict & target (N, 1, *) into one hot vector (N, C, *)
        predict_onehot = torch.zeros((N, C, last_size)).cuda() # (N, 1, *) ==> (N, C, *)
        predict_onehot.scatter_(1, predict, 1) # (N, C, *)
        target_onehot = torch.zeros((N, C, last_size)).cuda() # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1) # (N, C, *)

        true_positive = torch.sum(predict_onehot * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        total_predict = torch.sum(predict_onehot, dim=2)  # (N, C)
        ## Recall = TP / (TP + FN), precision = TP / (TP + TP)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)  # (N, C)
        precision = (true_positive + self.smooth) / (total_predict + self.smooth)  # (N, C)
        f1 = 2 * recall * precision / (recall + precision)

        if hasattr(self, 'weight'):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input.data)
                f1 = f1 * self.weight * C  # (N, C)
        f1_loss = 1 - torch.mean(f1)  # 1

        return f1_loss

if __name__ == '__main__':
    y_target = torch.Tensor([[0, 1], [1, 0]]).long().cuda(0)
    y_predict = torch.Tensor([[[1.5, 1.0], [2.8, 1.6]],
                           [[1.0, 1.0], [2.4, 0.3]]]
                          ).cuda(0)

    criterion = F1Loss(weight=[1, 1])
    loss = criterion(y_predict, y_target)
    print(loss)