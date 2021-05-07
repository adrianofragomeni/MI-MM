import torch as th
import torch.nn.functional as F

class MaxMarginRankingLoss(th.nn.Module):

    def __init__(self, margin=1, fix_norm=True):
        super(MaxMarginRankingLoss,self).__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)

        x2 = x.contiguous().view(-1, 1)

        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0]) 
            keep1 = keep.view(-1, 1)
            keep_idx = th.nonzero(keep1.flatten(),as_tuple=False).flatten()

            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()
