import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):


    def __init__(self, temperature=0.1):
        super(KLLoss, self).__init__()
        self.temperature = temperature


    def forward(self, p, q, reduce='mean'):
        """
        计算KL divergence loss
        p: [N, C]
        q: [N, C]
        """
        # 转换成log probabilities
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        # 计算损失
        loss_func = torch.nn.KLDivLoss(size_average=False, reduce=False)
        loss_pq = loss_func(p.log(), q)
        loss_qp = loss_func(q.log(), p)

        if reduce == 'sum':
            loss_pq = loss_pq.sum()
            loss_qp = loss_qp.sum()
        else:
            loss_pq = loss_pq.mean()
            loss_qp = loss_qp.mean()
        loss = (loss_pq + loss_qp) / 2
        return loss