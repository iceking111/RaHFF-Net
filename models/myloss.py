import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):

    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
    def forward(self, input, target, weight=None, reduction='mean',ignore_index=255):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=weight,ignore_index=ignore_index, reduction=reduction)




if __name__ == '__main__':
    net = CombinedLoss()
    input = torch.tensor([[0.3, 0.2, 0.6], [0.8, 0.4, 0.9]])
    target = torch.tensor([[0, 0, 0], [1, 0, 1]])

    loss = net(input, target)
    print(loss)

