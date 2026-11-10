"""Matched fixed-feature readers and their actual float32 serving maps."""
import torch
from torch import nn
from torch.nn import functional as F


class RecoveryReader(nn.Module):
    def __init__(self,width,classes,architecture,mean,sd,initial):
        super().__init__();self.architecture=architecture
        self.register_buffer('mean',mean.reshape(1,-1).float())
        self.register_buffer('sd',sd.reshape(1,-1).float())
        self.linear=nn.Linear(width,classes)
        with torch.no_grad():
            self.linear.weight.copy_((initial['weight'].double()*sd.reshape(-1)).float())
            self.linear.bias.copy_((initial['bias'].double()+initial['weight'].double()@mean.reshape(-1)).float())
        if architecture=='quadratic':self.quadratic=nn.Linear(width,classes,bias=False);nn.init.zeros_(self.quadratic.weight)
        elif architecture=='residual32':
            self.hidden=nn.Linear(width,32);self.output=nn.Linear(32,classes,bias=False);nn.init.zeros_(self.output.weight)
        elif architecture!='affine':raise ValueError(architecture)

    def forward(self,h):
        x=(h.float()-self.mean)/self.sd;z=self.linear(x)
        if self.architecture=='quadratic':z=z+self.quadratic(x.square()-1)
        if self.architecture=='residual32':z=z+self.output(F.relu(self.hidden(x)))
        return z


@torch.no_grad()
def export_reader(reader):
    mu,sd=reader.mean.double().reshape(-1),reader.sd.double().reshape(-1)
    w=reader.linear.weight.double()/sd;b=reader.linear.bias.double()-w@mu
    result=dict(kind=reader.architecture,weight=w.float(),bias=b.float())
    if reader.architecture=='quadratic':
        v=reader.quadratic.weight.double()/sd.square()
        result.update(weight=(w-2*v*mu).float(),bias=(b+(v*mu.square()).sum(1)-reader.quadratic.weight.double().sum(1)).float(),quadratic_weight=v.float())
    if reader.architecture=='residual32':
        h=reader.hidden.weight.double()/sd
        result.update(hidden_weight=h.float(),hidden_bias=(reader.hidden.bias.double()-h@mu).float(),
                      output_weight=reader.output.weight.detach().clone())
    return result


def reader_logits(features,reader):
    h=features.float();z=F.linear(h,reader['weight'],reader['bias'])
    if reader['kind']=='quadratic':z=z+F.linear(h.square(),reader['quadratic_weight'])
    if reader['kind']=='residual32':z=z+F.linear(F.relu(F.linear(h,reader['hidden_weight'],reader['hidden_bias'])),reader['output_weight'])
    return z


def current_and_legacy(features,current,reader):
    """One shared first readout product; no second encoder or reference model."""
    h=features.float();weights=[current.weight,reader['weight']];biases=[current.bias,reader['bias']]
    if reader['kind']=='residual32':weights.append(reader['hidden_weight']);biases.append(reader['hidden_bias'])
    joined=F.linear(h,torch.cat(weights),torch.cat(biases));c=current.out_features;d=len(reader['bias'])
    now,old=joined[:,:c],joined[:,c:c+d]
    if reader['kind']=='residual32':old=old+F.linear(F.relu(joined[:,c+d:]),reader['output_weight'])
    if reader['kind']=='quadratic':old=old+F.linear(h.square(),reader['quadratic_weight'])
    return now,old
