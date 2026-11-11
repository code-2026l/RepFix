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

class WholeAMPTwin(nn.Module):
    def __init__(self,prefix,old,current,reader,legacy):
        super().__init__();self.prefix=prefix;self.old=old;self.current=current
        for name,value in [('weight',reader['weight']),('bias',reader['bias']),('old_weight',legacy['weight']),('old_bias',legacy['bias'])]:self.register_buffer(name,value.detach().clone())
    def forward(self,x):
        with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
            h=self.prefix(x);current=self.current(h);old=self.old(h)
        with torch.autocast('cpu',enabled=False):return F.linear(current.float(),self.weight,self.bias),F.linear(old.float(),self.old_weight,self.old_bias)
