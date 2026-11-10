"""Final-block attention with one CLS query, shared by both serving baselines."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from nonlinear_suffix import AffineMap

class TokenPrefix(nn.Module):
    def __init__(self,model,blocks=1):
        super().__init__();self.conv=model.conv_proj;self.class_token=model.class_token
        self.position=model.encoder.pos_embedding;self.dropout=model.encoder.dropout
        self.layers=nn.Sequential(*list(model.encoder.layers.children())[:-blocks])
    def forward(self,x):
        h=self.conv(x).flatten(2).transpose(1,2)
        return self.layers(self.dropout(torch.cat([self.class_token.expand(h.shape[0],-1,-1),h],1)+self.position))

class TokenSuffix(nn.Module):
    def __init__(self,block,final_norm,preceding=()):
        super().__init__();attention=block.self_attention;d=attention.embed_dim;self.heads=attention.num_heads
        self.preceding=nn.Sequential(*[copy.deepcopy(layer) for layer in preceding])
        qkv=nn.Linear(d,3*d,dtype=attention.in_proj_weight.dtype,device=attention.in_proj_weight.device)
        qkv.weight.data.copy_(attention.in_proj_weight);qkv.bias.data.copy_(attention.in_proj_bias)
        self.maps=nn.ModuleList([AffineMap(qkv),AffineMap(attention.out_proj),AffineMap(block.mlp[0]),AffineMap(block.mlp[3])])
        self.norm1=copy.deepcopy(block.ln_1).requires_grad_(False)
        self.ln_mlp=copy.deepcopy(block.ln_2).requires_grad_(False)
        self.norm2=copy.deepcopy(final_norm).requires_grad_(False)
    def attention(self,x):
        x=self.preceding(x)
        q,k,v=self.maps[0](self.norm1(x)).chunk(3,dim=-1);b,n,d=k.shape;head=d//self.heads
        q=q[:,:1].reshape(b,1,self.heads,head).transpose(1,2)
        k=k.reshape(b,n,self.heads,head).transpose(1,2);v=v.reshape(b,n,self.heads,head).transpose(1,2)
        h=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,1,d)
        return x[:,0]+self.maps[1](h[:,0])
    def hidden(self,x):return F.gelu(self.maps[2](self.ln_mlp(self.attention(x))))
    def features(self,x):
        h=self.attention(x);hidden=F.gelu(self.maps[2](self.ln_mlp(h)))
        return self.norm2(h+self.maps[3](hidden)),hidden
    def forward(self,x):return self.features(x)[0]
    def dense(self):return copy.deepcopy(self)

def build_token(blocks=1):
    from torchvision.models import vit_b_32,ViT_B_32_Weights
    model=vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).eval()
    earlier=list(model.encoder.layers.children())[-blocks:-1]
    return TokenPrefix(model,blocks).eval().requires_grad_(False),TokenSuffix(model.encoder.layers[-1],model.encoder.ln,earlier).eval()
