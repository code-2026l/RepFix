"""Shared frozen prefixes and versioned nonlinear suffixes.

Late tuning, low-rank adaptation and side heads are established constructions.
These modules make their two-output contract and fair merged comparator explicit.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class CNNPrefix(nn.Module):
    def __init__(self,model):
        super().__init__();block=model.layer4[-1]
        self.body=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,*list(model.layer4.children())[:-1])
        if hasattr(block,'conv3'):self.pre=nn.Sequential(block.conv1,block.bn1,block.relu,block.conv2,block.bn2,copy.deepcopy(block.relu))
        else:self.pre=nn.Sequential(block.conv1,block.bn1,block.relu)
        self.requires_grad_(False);self.eval()
    def forward(self,x):
        residual=self.body(x);return self.pre(residual).contiguous(),residual.contiguous()


class ViTPrefix(nn.Module):
    def __init__(self,model):
        super().__init__();last=model.encoder.layers[-1]
        self.conv=model.conv_proj;self.class_token=model.class_token;self.position=model.encoder.pos_embedding;self.dropout=model.encoder.dropout
        self.layers=nn.Sequential(*list(model.encoder.layers.children())[:-1])
        self.ln1=last.ln_1;self.attention=last.self_attention;self.attention_drop=last.dropout;self.ln2=last.ln_2
        self.mlp_pre=nn.Sequential(*list(last.mlp.children())[:3]);self.requires_grad_(False);self.eval()
    def forward(self,x):
        h=self.conv(x);h=h.flatten(2).transpose(1,2);h=torch.cat([self.class_token.expand(h.shape[0],-1,-1),h],1)
        h=self.layers(self.dropout(h+self.position));y=self.ln1(h);attention=self.attention(y,y,y,need_weights=False)[0]
        residual=h+self.attention_drop(attention);cls=residual[:,0]
        return self.mlp_pre(self.ln2(cls)),cls


def build_backbone(arch,pretrained=True):
    from torchvision.models import resnet18,resnet50,vit_b_32,ResNet18_Weights,ResNet50_Weights,ViT_B_32_Weights
    specs={'resnet18':(resnet18,ResNet18_Weights.IMAGENET1K_V1),'resnet50':(resnet50,ResNet50_Weights.IMAGENET1K_V2),'vit_b32':(vit_b_32,ViT_B_32_Weights.IMAGENET1K_V1)}
    constructor,weights=specs[arch];model=constructor(weights=weights if pretrained else None).eval()
    count=sum(p.numel() for p in model.parameters())
    if arch.startswith('resnet'):
        model.to(memory_format=torch.channels_last);block=model.layer4[-1];assert block.downsample is None
        tail=fuse_conv_bn_eval(block.conv3,block.bn3) if arch=='resnet50' else fuse_conv_bn_eval(block.conv2,block.bn2)
        prefix=CNNPrefix(model);norm=nn.Identity()
    else:prefix=ViTPrefix(model);tail=copy.deepcopy(model.encoder.layers[-1].mlp[3]);norm=copy.deepcopy(model.encoder.ln)
    prefix.eval().requires_grad_(False);tail.eval().requires_grad_(False);norm.eval().requires_grad_(False)
    return prefix,tail,norm,dict(pretrained_parameters=count,weights_url=weights.url,image_size=224 if arch=='vit_b32' else 64)


def finish(value,residual,norm):
    if value.ndim==4:return F.relu(value+residual).mean((-2,-1))
    return norm(value+residual)


class LateDelta(nn.Module):
    def __init__(self,tail,norm,right):
        super().__init__();self.tail=copy.deepcopy(tail).requires_grad_(False);self.norm=copy.deepcopy(norm).requires_grad_(False)
        self.convolution=isinstance(tail,nn.Conv2d);r=len(right);out=tail.weight.shape[0]
        self.u=nn.Parameter(torch.zeros(out,r,dtype=tail.weight.dtype));self.v=nn.Parameter(right.clone().to(tail.weight.dtype))
        self.bias=nn.Parameter(torch.zeros(out,dtype=tail.weight.dtype))
    def correction(self,x):
        if self.convolution:
            kernel=self.v.reshape(len(self.v),*self.tail.weight.shape[1:]);h=F.conv2d(x,kernel,padding=self.tail.padding)
            return F.conv2d(h,self.u[:,:,None,None],self.bias)
        return F.linear(F.linear(x,self.v),self.u,self.bias)
    def forward(self,x,residual,base=None):
        old=self.tail(x) if base is None else base
        return finish(old+self.correction(x),residual,self.norm)
    @torch.no_grad()
    def merged(self):
        tail=copy.deepcopy(self.tail);tail.weight.add_((self.u@self.v).reshape_as(tail.weight));tail.bias.add_(self.bias)
        return tail


class DenseSuffix(nn.Module):
    def __init__(self,tail,norm):
        super().__init__();self.tail=copy.deepcopy(tail).requires_grad_(True);self.norm=copy.deepcopy(norm).requires_grad_(False)
    def forward(self,x,residual,base=None):return finish(self.tail(x),residual,self.norm)


class ResidualHead(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=copy.deepcopy(base);d=base.linear.in_features;c=base.linear.out_features
        self.hidden=nn.Linear(d,128);self.out=nn.Linear(128,c)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,h):
        x=(h-self.base.mean)/self.base.sd
        return self.base.linear(x)+self.out(F.gelu(self.hidden(x)))
    @torch.no_grad()
    def serving(self):
        h=copy.deepcopy(self.hidden);w=h.weight/self.base.sd
        h.weight.copy_(w);h.bias.sub_(w@self.base.mean)
        return self.base.affine(),h,copy.deepcopy(self.out)


class VisionServing(nn.Module):
    """Both methods receive the same prefix, suffix fusion and CLS pruning."""
    def __init__(self,prefix,tail,norm,head,legacy,update=None,both=False,independent_prefix=False):
        super().__init__();self.prefix=prefix;self.tail=tail;self.norm=norm;self.update=update;self.both=both
        self.register_buffer('weight',head['weight']);self.register_buffer('bias',head['bias'])
        self.register_buffer('old_weight',legacy['weight']);self.register_buffer('old_bias',legacy['bias'])
        self.independent_prefix=copy.deepcopy(prefix) if independent_prefix else None
    def forward(self,images):
        x,residual=self.prefix(images);base=self.tail(x)
        if self.update is not None:current=self.update(x,residual,base)
        else:current=finish(base,residual,self.norm)
        own=F.linear(current,self.weight,self.bias)
        if not self.both:return own
        if self.independent_prefix is not None:
            old_x,old_residual=self.independent_prefix(images);old=finish(self.tail(old_x),old_residual,self.norm)
        else:old=finish(base,residual,self.norm)
        return own,F.linear(old,self.old_weight,self.old_bias)


class IndependentServing(nn.Module):
    def __init__(self,current,healthy):
        super().__init__();self.current=current;self.healthy=healthy
    def forward(self,images):return self.current(images),self.healthy(images)
