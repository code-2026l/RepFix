"""Standard CPU autocast encoder with an explicitly FP32 consumer."""
import torch
from torch import nn
from torch.nn import functional as F

class WholeAMPReadout(nn.Module):
    def __init__(self,prefix,suffix,reader):
        super().__init__();self.prefix=prefix;self.suffix=suffix
        self.register_buffer('weight',reader['weight'].detach().clone());self.register_buffer('bias',reader['bias'].detach().clone())
    def forward(self,x):
        with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):h=self.suffix(self.prefix(x))
        with torch.autocast('cpu',enabled=False):return F.linear(h.float(),self.weight,self.bias)
