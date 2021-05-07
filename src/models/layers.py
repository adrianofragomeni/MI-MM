import torch.nn as nn
import torch.nn.functional as F
import torch as th

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True, normalize=True):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
        self.gating = gating
        self.normalize= normalize
  
    def forward(self,x):
        
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        if self.normalize:
            x = F.normalize(x)

        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1) 

        x = th.cat((x, x1), 1)
        
        return F.glu(x,1)
    