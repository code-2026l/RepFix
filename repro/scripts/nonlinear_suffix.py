"""Complete nonlinear suffixes with shared frozen prefixes; standard controls."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from late_replay import build_backbone,ViTPrefix

class AttentionPrefix(ViTPrefix):
    def __init__(self,model):
        super().__init__(model);del self.mlp_pre;del self.ln2
    def forward(self,x):
        h=self.conv(x).flatten(2).transpose(1,2);h=torch.cat([self.class_token.expand(h.shape[0],-1,-1),h],1)
        h=self.layers(self.dropout(h+self.position));y=self.ln1(h)
        return (h+self.attention_drop(self.attention(y,y,y,need_weights=False)[0]))[:,0]

class AffineMap(nn.Module):
    def __init__(self,module,rank=None,right=None):
        super().__init__();self.convolution=isinstance(module,nn.Conv2d);self.padding=module.padding if self.convolution else (0,0)
        self.rank=0 if rank is None else rank
        if rank is None:
            self.weight=nn.Parameter(module.weight.detach().clone());self.bias=nn.Parameter(module.bias.detach().clone())
        else:
            self.register_buffer('weight',module.weight.detach().clone());self.register_buffer('bias',module.bias.detach().clone())
            self.u=nn.Parameter(torch.zeros(module.weight.shape[0],rank));self.v=nn.Parameter(right[:rank].clone().float());self.delta_bias=nn.Parameter(torch.zeros(len(module.bias)))
    def forward(self,x):
        if self.convolution:
            y=F.conv2d(x,self.weight,self.bias,padding=self.padding)
            if self.rank:y=y+F.conv2d(F.conv2d(x,self.v.reshape(self.rank,*self.weight.shape[1:]),padding=self.padding),self.u[:,:,None,None],self.delta_bias)
        else:
            y=F.linear(x,self.weight,self.bias)
            if self.rank:y=y+F.linear(F.linear(x,self.v),self.u,self.delta_bias)
        return y
    @torch.no_grad()
    def dense(self):
        result=copy.deepcopy(self)
        if self.rank:
            weight=self.weight+(self.u@self.v).reshape_as(self.weight);bias=self.bias+self.delta_bias
            result.rank=0;del result.u;del result.v;del result.delta_bias
            result.weight.copy_(weight);result.bias.copy_(bias)
        return result

class FullSuffix(nn.Module):
    def __init__(self,maps,norm1=None,norm2=None):
        super().__init__();self.maps=nn.ModuleList(maps);self.vit=norm1 is not None
        self.norm1=nn.Identity() if norm1 is None else copy.deepcopy(norm1).requires_grad_(False)
        self.norm2=nn.Identity() if norm2 is None else copy.deepcopy(norm2).requires_grad_(False)
    def hidden(self,x):
        if self.vit:return F.gelu(self.maps[0](self.norm1(x)))
        h=x
        for layer in self.maps[:-1]:h=F.relu(layer(h))
        return h
    def forward(self,x):
        h=self.hidden(x);value=self.maps[-1](h)
        if self.vit:return self.norm2(value+x)
        return F.relu(value+x).mean((-2,-1))
    @torch.no_grad()
    def dense(self):
        result=copy.deepcopy(self);result.maps=nn.ModuleList([m.dense() for m in self.maps]);return result
    def lowrank(self,rights,rank=64):
        modules=[]
        for old,right in zip(self.maps,rights):
            shape=old.weight.shape
            module=nn.Conv2d(shape[1],shape[0],shape[-1],padding=old.padding) if old.convolution else nn.Linear(shape[1],shape[0])
            module.weight.data.copy_(old.weight);module.bias.data.copy_(old.bias);modules.append(AffineMap(module,rank,right))
        return FullSuffix(modules,self.norm1 if self.vit else None,self.norm2 if self.vit else None)

def build_full(arch):
    from torchvision.models import resnet18,resnet50,vit_b_32,ResNet18_Weights,ResNet50_Weights,ViT_B_32_Weights
    constructor,weights={'resnet18':(resnet18,ResNet18_Weights.IMAGENET1K_V1),'resnet50':(resnet50,ResNet50_Weights.IMAGENET1K_V2),'vit_b32':(vit_b_32,ViT_B_32_Weights.IMAGENET1K_V1)}[arch]
    model=constructor(weights=weights).eval()
    if arch=='vit_b32':
        prefix=AttentionPrefix(model);last=model.encoder.layers[-1]
        suffix=FullSuffix([AffineMap(last.mlp[0]),AffineMap(last.mlp[3])],last.ln_2,model.encoder.ln)
    else:
        model.to(memory_format=torch.channels_last);b=model.layer4[-1]
        prefix=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,*list(model.layer4.children())[:-1])
        layers=[fuse_conv_bn_eval(b.conv1,b.bn1),fuse_conv_bn_eval(b.conv2,b.bn2)]
        if arch=='resnet50':layers.append(fuse_conv_bn_eval(b.conv3,b.bn3))
        suffix=FullSuffix([AffineMap(layer) for layer in layers])
    return prefix.eval().requires_grad_(False),suffix.eval()

class SuffixServing(nn.Module):
    def __init__(self,prefix,old,current,head,legacy,both=False,independent=False):
        super().__init__();self.prefix=prefix;self.old=old;self.current=current;self.both=both
        self.other_prefix=copy.deepcopy(prefix) if independent else None
        self.register_buffer('weight',head['weight']);self.register_buffer('bias',head['bias'])
        self.register_buffer('old_weight',legacy['weight']);self.register_buffer('old_bias',legacy['bias'])
    def forward(self,images):
        x=self.prefix(images);current=F.linear(self.current(x),self.weight,self.bias)
        if not self.both:return current
        old_x=x if self.other_prefix is None else self.other_prefix(images)
        return current,F.linear(self.old(old_x),self.old_weight,self.old_bias)
