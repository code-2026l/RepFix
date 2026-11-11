"""Mergeable sparse/low-rank updates with a shared healthy dense computation.

Low-rank adaptation and incremental computation are established ideas. This
module makes the two-version identity explicit and audits its numerical scope.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def linear_delta_states(x,pre1,pre2,weight2_columns,support,delta1,bias1,u,v,bias2):
    old1=F.relu(pre1);old2=F.relu(pre2)
    if len(support):
        selected=F.relu(pre1[:,support]+F.linear(x,delta1,bias1))
        changed=selected-old1[:,support]
        current1=old1.index_copy(1,support,selected)
        correction=F.linear(changed,weight2_columns)
    else:current1=old1;correction=torch.zeros_like(pre2)
    current_pre2=pre2+correction+F.linear(F.linear(current1,v),u)+bias2
    return F.relu(current_pre2),old2


def convolution_delta_states(x,pre1,pre2,weight2_columns,support,delta1,bias1,u,v,bias2):
    """Identity-residual block with fixed/fused normalization and 3x3 kernels."""
    old1=F.relu(pre1)
    if len(support):
        selected=F.relu(pre1[:,support]+F.conv2d(x,delta1,bias1,padding=1))
        changed=selected-old1[:,support];current1=old1.index_copy(1,support,selected)
        correction=F.conv2d(changed,weight2_columns,padding=1)
    else:current1=old1;correction=torch.zeros_like(pre2)
    current_pre2=pre2+correction+F.conv2d(F.conv2d(current1,v,padding=1),u)+bias2.reshape(1,-1,1,1)
    return F.relu(current_pre2+x),F.relu(pre2+x)


class SharedReferenceMLP(nn.Module):
    def __init__(self,healthy,support,right_vectors,A,B):
        super().__init__();self.first=copy.deepcopy(healthy.encoder[0]).requires_grad_(False)
        self.second=copy.deepcopy(healthy.encoder[2]).requires_grad_(False)
        self.final=copy.deepcopy(healthy.encoder[4]);self.primary=copy.deepcopy(healthy.primary);self.aux=copy.deepcopy(healthy.aux)
        self.healthy_final=copy.deepcopy(healthy.encoder[4]).requires_grad_(False)
        self.register_buffer('support',support.clone().long());m=self.second.out_features;k=len(support)
        self.delta1=nn.Parameter(torch.zeros(k,self.first.in_features,dtype=self.first.weight.dtype))
        self.delta1_bias=nn.Parameter(torch.zeros(k,dtype=self.first.weight.dtype))
        self.u=nn.Parameter(torch.zeros(m,len(right_vectors),dtype=self.first.weight.dtype))
        self.v=nn.Parameter(right_vectors.clone().to(self.first.weight.dtype));self.delta2_bias=nn.Parameter(torch.zeros(m,dtype=self.first.weight.dtype))
        self.register_buffer('old_A',A.detach().double().clone());self.register_buffer('old_B',B.detach().double().clone())

    def old_parameters(self):
        return [p for layer in [self.first,self.second,self.final,self.primary,self.aux] for p in layer.parameters()]

    def update_parameters(self):return [self.delta1,self.delta1_bias,self.u,self.v,self.delta2_bias]

    def forward_states(self,x,cache=None):
        if cache is None:
            pre1=self.first(x);pre2=self.second(F.relu(pre1))
        else:pre1,pre2=cache
        return linear_delta_states(x,pre1,pre2,self.second.weight[:,self.support],self.support,
            self.delta1,self.delta1_bias,self.u,self.v,self.delta2_bias)

    def forward(self,x,cache=None):return self.final(self.forward_states(x,cache)[0])

    def legacy_logits(self,old2,compressed=False):
        if not compressed:return F.linear(self.healthy_final(old2).double(),self.old_A,self.old_B)
        weight=self.old_A@self.healthy_final.weight.double()
        bias=self.old_B+self.old_A@self.healthy_final.bias.double()
        return F.linear(old2.double(),weight,bias)

    @torch.no_grad()
    def merged(self):
        first=copy.deepcopy(self.first);second=copy.deepcopy(self.second)
        first.weight[self.support]+=self.delta1;first.bias[self.support]+=self.delta1_bias
        second.weight+=self.u@self.v;second.bias+=self.delta2_bias
        return nn.Sequential(first,nn.ReLU(),second,nn.ReLU(),copy.deepcopy(self.final))

    @torch.no_grad()
    def serving(self):
        current_weight=(self.primary.weight.double()@self.final.weight.double()).float()
        current_bias=(self.primary.bias.double()+self.primary.weight.double()@self.final.bias.double()).float()
        old_weight=self.old_A@self.healthy_final.weight.double();old_bias=self.old_B+self.old_A@self.healthy_final.bias.double()
        return dict(first_weight=self.first.weight.detach().clone(),first_bias=self.first.bias.detach().clone(),
            second_weight=self.second.weight.detach().clone(),second_bias=self.second.bias.detach().clone(),
            support=self.support.clone(),delta1=self.delta1.detach().clone(),delta1_bias=self.delta1_bias.detach().clone(),
            columns=self.second.weight[:,self.support].detach().clone(),u=self.u.detach().clone(),v=self.v.detach().clone(),
            delta2_bias=self.delta2_bias.detach().clone(),current_weight=current_weight,current_bias=current_bias,
            old_weight=old_weight.float(),old_bias=old_bias.float())


def serve_mlp(x,p):
    pre1=F.linear(x,p['first_weight'],p['first_bias']);pre2=F.linear(F.relu(pre1),p['second_weight'],p['second_bias'])
    current,old=linear_delta_states(x,pre1,pre2,p['columns'],p['support'],p['delta1'],p['delta1_bias'],p['u'],p['v'],p['delta2_bias'])
    return F.linear(current,p['current_weight'],p['current_bias']),F.linear(old,p['old_weight'],p['old_bias'])


@torch.no_grad()
def merged_serving(p):
    first=p['first_weight'].clone();bias=p['first_bias'].clone()
    first[p['support']]+=p['delta1'];bias[p['support']]+=p['delta1_bias']
    return dict(first_weight=first,first_bias=bias,second_weight=p['second_weight']+p['u']@p['v'],
        second_bias=p['second_bias']+p['delta2_bias'],current_weight=p['current_weight'],current_bias=p['current_bias'])


def serve_merged_mlp(x,p):
    h=F.relu(F.linear(x,p['first_weight'],p['first_bias']));h=F.relu(F.linear(h,p['second_weight'],p['second_bias']))
    return F.linear(h,p['current_weight'],p['current_bias'])
