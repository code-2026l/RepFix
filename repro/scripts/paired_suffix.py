"""Exact grouped dense evaluation of one or two suffix versions.

The one-version implementation is a competing current-only baseline. Packing
uses established concatenated/grouped products; no new learning rule is added.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class PackedConvolution(nn.Module):
    def __init__(self,maps,first,low_precision=False):
        super().__init__();self.padding=maps[0].padding;self.groups=1 if first else len(maps);self.low_precision=low_precision
        weight=torch.cat([m.weight.detach() for m in maps]);self.register_buffer('weight',weight.bfloat16() if low_precision else weight)
        self.register_buffer('bias',torch.cat([m.bias.detach() for m in maps]))
    def forward(self,x):
        if self.low_precision:return F.conv2d(x.bfloat16(),self.weight,None,padding=self.padding,groups=self.groups).float()+self.bias[None,:,None,None]
        return F.conv2d(x,self.weight,self.bias,padding=self.padding,groups=self.groups)

class PackedCNN(nn.Module):
    def __init__(self,old,current,reader,legacy,both,low_precision=False):
        super().__init__();self.versions=2 if both else 1;models=[old,current] if both else [current]
        self.layers=nn.ModuleList([PackedConvolution([m.maps[i] for m in models],i==0,low_precision) for i in range(len(old.maps))]);self.both=both
        for name,value in [('weight',reader['weight']),('bias',reader['bias']),('old_weight',legacy['weight']),('old_bias',legacy['bias'])]:self.register_buffer(name,value.detach().clone())
    def forward(self,x):
        h=x
        for layer in self.layers[:-1]:h=F.relu(layer(h))
        shortcut=torch.cat([x]*self.versions,1)
        h=F.relu(self.layers[-1](h)+shortcut).mean((-2,-1)).reshape(x.shape[0],self.versions,-1)
        z=F.linear(h[:,-1],self.weight,self.bias)
        if self.both:return z,F.linear(h[:,0],self.old_weight,self.old_bias)
        return z

def descriptors(suffix):
    blocks=[]
    for b in suffix.preceding:
        blocks.append((b.ln_1,b.ln_2,b.self_attention.num_heads,[
            b.self_attention.in_proj_weight,b.self_attention.in_proj_bias,b.self_attention.out_proj.weight,b.self_attention.out_proj.bias,
            b.mlp[0].weight,b.mlp[0].bias,b.mlp[3].weight,b.mlp[3].bias]))
    blocks.append((suffix.norm1,suffix.ln_mlp,suffix.heads,[t for m in suffix.maps for t in [m.weight,m.bias]]))
    return blocks

class PackedAttentionBlock(nn.Module):
    def __init__(self,blocks,first,last,low_precision=False):
        super().__init__();self.first=first;self.last=last;self.versions=len(blocks);self.heads=blocks[0][2];self.low_precision=low_precision
        self.ln1=copy.deepcopy(blocks[0][0]);self.ln2=copy.deepcopy(blocks[0][1])
        for block in blocks[1:]:
            for i in [0,1]:
                assert block[i].eps==blocks[0][i].eps
                assert all(torch.equal(v,block[i].state_dict()[k]) for k,v in blocks[0][i].state_dict().items())
        for i,name in enumerate(['qkv_weight','qkv_bias','out_weight','out_bias','fc1_weight','fc1_bias','fc2_weight','fc2_bias']):
            value=torch.stack([b[3][i].detach() for b in blocks]);self.register_buffer(name,value.bfloat16() if low_precision and name.endswith('weight') else value)
        weight=torch.cat([b[3][0].detach() for b in blocks]);self.register_buffer('qkv_cat_weight',weight.bfloat16() if low_precision else weight)
        self.register_buffer('qkv_cat_bias',torch.cat([b[3][1].detach() for b in blocks]))
    def project(self,x,weight,bias):
        v,b,n,d=x.shape
        x=x.reshape(v,b*n,d)
        value=torch.bmm(x.bfloat16() if self.low_precision else x,weight.transpose(1,2))
        if self.low_precision:value=value.float()
        return (value+bias[:,None,:]).reshape(v,b,n,weight.shape[1])
    def forward(self,x):
        if self.first:
            b,n,d=x.shape;residual=x.unsqueeze(0).expand(self.versions,-1,-1,-1)
            y=self.ln1(x)
            if self.low_precision:y=F.linear(y.bfloat16(),self.qkv_cat_weight).float()+self.qkv_cat_bias
            else:y=F.linear(y,self.qkv_cat_weight,self.qkv_cat_bias)
            qkv=y.reshape(b,n,self.versions,-1).permute(2,0,1,3)
        else:
            residual=x;qkv=self.project(self.ln1(x),self.qkv_weight,self.qkv_bias)
        v,b,n,three_d=qkv.shape;d=three_d//3;head=d//self.heads;q,k,value=qkv.chunk(3,dim=-1)
        if self.last:q=q[:,:,:1];residual=residual[:,:,:1]
        nq=q.shape[2]
        q=q.reshape(v*b,nq,self.heads,head).transpose(1,2)
        k=k.reshape(v*b,n,self.heads,head).transpose(1,2);value=value.reshape(v*b,n,self.heads,head).transpose(1,2)
        h=F.scaled_dot_product_attention(q,k,value).transpose(1,2).reshape(v,b,nq,d)
        h=residual+self.project(h,self.out_weight,self.out_bias)
        return h+self.project(F.gelu(self.project(self.ln2(h),self.fc1_weight,self.fc1_bias)),self.fc2_weight,self.fc2_bias)

class PackedTransformer(nn.Module):
    def __init__(self,old,current,reader,legacy,both,low_precision=False):
        super().__init__();self.both=both;models=[old,current] if both else [current];dd=[descriptors(m) for m in models];n=len(dd[0])
        self.layers=nn.ModuleList([PackedAttentionBlock([d[i] for d in dd],i==0,i==n-1,low_precision) for i in range(n)]);self.norm=copy.deepcopy(old.norm2)
        assert all(torch.equal(v,current.norm2.state_dict()[k]) for k,v in old.norm2.state_dict().items())
        for name,value in [('weight',reader['weight']),('bias',reader['bias']),('old_weight',legacy['weight']),('old_bias',legacy['bias'])]:self.register_buffer(name,value.detach().clone())
    def forward(self,x):
        for layer in self.layers:x=layer(x)
        h=self.norm(x[:,:,0]);z=F.linear(h[-1],self.weight,self.bias)
        if self.both:return z,F.linear(h[0],self.old_weight,self.old_bias)
        return z

def pack_suffix(old,current,reader,legacy,both=True,low_precision=False):
    assert not (low_precision and both),'Reference-only low precision must not change the current version'
    constructor=PackedTransformer if hasattr(old,'preceding') else PackedCNN
    return constructor(old,current,reader,legacy,both,low_precision).eval().requires_grad_(False)
