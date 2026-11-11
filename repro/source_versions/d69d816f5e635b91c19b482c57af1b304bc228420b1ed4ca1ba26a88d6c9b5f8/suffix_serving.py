"""Matched graph forms and original-feature controls for two-output serving."""
import torch
from torch import nn
from torch.nn import functional as F

class SuffixReadout(nn.Module):
    def __init__(self,old,current,reader,legacy,mode='U',both=False,hidden=None,mean=None,sd=None):
        super().__init__();self.old=old;self.current=current;self.mode=mode;self.both=both;self.hidden=hidden
        for name,value in [('weight',reader['weight']),('bias',reader['bias']),('old_weight',legacy['weight']),('old_bias',legacy['bias']),('mean',mean),('sd',sd)]:self.register_buffer(name,value)
    def forward(self,x):
        old=None
        if self.mode=='H':
            old,hidden=self.old.features(x);features=torch.cat([old,hidden],1)
        elif self.mode=='N':
            old=self.old(x);features=torch.cat([old,F.gelu(self.hidden((old-self.mean)/self.sd))],1)
        elif self.mode=='F':old=self.old(x);features=old
        else:features=self.current(x)
        z=F.linear(features,self.weight,self.bias)
        if not self.both:return z
        if old is None:old=self.old(x)
        return z,F.linear(old,self.old_weight,self.old_bias)

class WholeNetwork(nn.Module):
    def __init__(self,prefix,suffix):super().__init__();self.prefix=prefix;self.suffix=suffix
    def forward(self,x):return self.suffix(self.prefix(x))
