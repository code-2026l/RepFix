"""Current FP32 function with a separately represented reference product path."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_suffix import pack_suffix

class MixedReference(nn.Module):
    def __init__(self,old,current,reader,legacy,low_precision):
        super().__init__();self.current=current;self.old=pack_suffix(old,old,legacy,legacy,False,low_precision)
        self.register_buffer('weight',reader['weight'].detach().clone());self.register_buffer('bias',reader['bias'].detach().clone())
    def forward(self,x):return F.linear(self.current(x),self.weight,self.bias),self.old(x)
