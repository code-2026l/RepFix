"""Two versions of an identity residual block from one healthy trace."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from shared_reference_replay import convolution_delta_states


class SharedReferenceBlock(nn.Module):
    def __init__(self,first,second,support,right,head):
        super().__init__();self.first=copy.deepcopy(first).requires_grad_(False);self.second=copy.deepcopy(second).requires_grad_(False)
        self.register_buffer('support',support.clone().long());c=second.out_channels;k=len(support);r=len(right)
        self.delta1=nn.Parameter(torch.zeros(k,c,3,3));self.delta1_bias=nn.Parameter(torch.zeros(k))
        self.u=nn.Parameter(torch.zeros(c,r,1,1));self.v=nn.Parameter(right.reshape(r,c,3,3).float().clone());self.delta2_bias=nn.Parameter(torch.zeros(c))
        self.head=copy.deepcopy(head)

    def forward_states(self,x,cached=None):
        p1=self.first(x) if cached is None else cached[0]
        p2=self.second(F.relu(p1)) if cached is None else cached[1]
        return convolution_delta_states(x,p1,p2,self.second.weight[:,self.support],self.support,
            self.delta1,self.delta1_bias,self.u,self.v,self.delta2_bias)

    def forward(self,x,cached=None):
        current,_=self.forward_states(x,cached);return self.head(current.mean((-2,-1)))

    @torch.no_grad()
    def merged(self):
        first=copy.deepcopy(self.first);second=copy.deepcopy(self.second)
        first.weight[self.support]+=self.delta1;first.bias[self.support]+=self.delta1_bias
        second.weight+=torch.einsum('or,rijk->oijk',self.u[:,:,0,0],self.v);second.bias+=self.delta2_bias
        return first,second


class NormalizedHead(nn.Module):
    def __init__(self,weight,bias,mean,sd):
        super().__init__();self.linear=nn.Linear(weight.shape[1],len(bias));self.linear.weight.data.copy_(weight);self.linear.bias.data.copy_(bias)
        self.register_buffer('mean',mean.float().reshape(-1));self.register_buffer('sd',sd.float().reshape(-1))
    def forward(self,h):return self.linear((h-self.mean)/self.sd)

    @torch.no_grad()
    def affine(self):
        weight=self.linear.weight/self.sd
        return dict(weight=weight.clone(),bias=(self.linear.bias-weight@self.mean).clone())


def dense_block(x,first,second):return F.relu(second(F.relu(first(x)))+x)
