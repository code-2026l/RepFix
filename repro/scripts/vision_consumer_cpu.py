"""Gated fine-task updates with a distinct coarse-task ResNet-18 consumer.

Only the official CIFAR-100 training pickle is decoded during the screen.
The extra head/map is trained using cached features/logits, never coarse labels.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
import paired_readout_continuation as paired
from readout_transfer_audit import fit_head,stratified_split
from legacy_head_fit import fit_legacy_heads
from consumer_contracts import confidence_thresholds,evaluate_contracts
from vision_chart_cpu import preprocess,freeze_bn,bn_state

SOURCES=['vision_consumer_cpu.py','consumer_contracts.py','legacy_head_fit.py',
         'vision_chart_cpu.py','readout_transfer_audit.py','paired_readout_continuation.py']

def load_train():
    path=Path(os.environ['REPFIX_DATA_X'])/'v037/cifar-100-python/train'
    # This fixed cached pickle was extracted from the verified official archive.
    with path.open('rb') as stream:d=pickle.load(stream,encoding='bytes')
    images=torch.from_numpy(d[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).copy())
    return images,torch.tensor(d[b'fine_labels']),torch.tensor(d[b'coarse_labels']),hashlib.sha256(path.read_bytes()).hexdigest()

@torch.no_grad()
def features(model,images,indices):
    model.eval()
    return torch.cat([model(preprocess(images[indices[j:j+128]])).float()
                      for j in range(0,len(indices),128)])

def affine(head,mu,sd):
    weight=head.weight.detach().double()/sd.reshape(-1)
    bias=head.bias.detach().double()-weight@mu.reshape(-1)
    return dict(weight=weight.float(),bias=bias.float())

def linear(h,reader):return F.linear(h.float(),reader['weight'].float(),reader['bias'].float())

def encoder():
    m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);m.fc=nn.Identity()
    return m.to(memory_format=torch.channels_last).eval()

def header(a,seed):
    return dict(complete=False,device='cpu',cuda_visibility='',phase='development',seed=seed,
        plan_sha256=a.plan_sha256,source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES},
        splits=dict(official_test_loaded=False,test_metrics_computed=False),branches=[])

def prepare(a):
    start=time.time();torch.manual_seed(a.seed);np.random.seed(a.seed)
    images,fine,coarse,sha=load_train();pool,val=stratified_split(fine,.1,a.seed)
    perm=torch.randperm(len(pool),generator=torch.Generator().manual_seed(a.seed+31000))
    anchor,fit=pool[perm[:2048]],pool[perm[2048:]]
    assert len(fit)==42952 and len(val)==5000
    mask=torch.rand(len(fit),generator=torch.Generator().manual_seed(a.seed+501))<.2
    model=encoder();hfit=features(model,images,fit);hval=features(model,images,val);ha=features(model,images,anchor)
    heads={}
    for name,labels in [('fine',fine),('coarse',coarse)]:
        clf,mu,sd=fit_head(hfit[mask].double(),labels[fit][mask]);heads[name]=affine(clf,mu,sd)
    reader=heads['coarse'];thresholds=confidence_thresholds(linear(hfit,reader))
    state=dict(encoder=model.state_dict(),heads=heads,fitting=fit,validation=val,anchors=anchor,labeled=mask,
        healthy_fit=hfit,healthy_validation=hval,healthy_anchors=ha,thresholds=thresholds,
        seed=a.seed,plan_sha256=a.plan_sha256,data_sha256=sha)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    torch.save(state,folder/'reference.pt');out=header(a,a.seed)
    out.update(complete=True,elapsed_s=time.time()-start,data_sha256=sha,
        reference_encoder_sha256=paired.state_hash(state['encoder']),reference_bn_sha256=paired.state_hash(bn_state(model)),
        encoder_parameters=sum(p.numel() for p in model.parameters()),
        splits=dict(fitting=len(fit),validation=len(val),anchors=len(anchor),labeled=int(mask.sum()),
            fitting_sha256=paired.state_hash(fit),validation_sha256=paired.state_hash(val),
            anchor_sha256=paired.state_hash(anchor),mask_sha256=paired.state_hash(mask),
            official_test_loaded=False,test_metrics_computed=False),
        healthy=dict(fine=paired.metrics(linear(hval,heads['fine']),fine[val]),
            coarse=evaluate_contracts(linear(hval,reader),linear(hval,reader),coarse[val],thresholds)))
    paired.save_json(a.out,out)

def branch(a):
    start=time.time();ref=torch.load(a.cache,map_location='cpu',weights_only=True)
    assert ref['plan_sha256']==a.plan_sha256
    seed=ref['seed'];torch.manual_seed(seed);np.random.seed(seed)
    images,fine,coarse,sha=load_train();assert sha==ref['data_sha256']
    fit,val,mask=ref['fitting'],ref['validation'],ref['labeled']
    model=encoder();model.load_state_dict(ref['encoder']);bn_sha=paired.state_hash(bn_state(model))
    primary=nn.Linear(512,100);primary.load_state_dict(ref['heads']['fine'])
    extra=None
    if a.mode=='P':extra=nn.Linear(512,20);extra.load_state_dict(ref['heads']['coarse'])
    if a.mode=='B':
        extra=nn.Linear(512,512,bias=False)
        with torch.no_grad():extra.weight.copy_(torch.eye(512))
    groups=[dict(params=primary.parameters(),lr=.01)]
    if a.mode!='F':groups.append(dict(params=model.parameters(),lr=.001))
    if extra is not None:groups.append(dict(params=extra.parameters(),lr=.01))
    opt=torch.optim.SGD(groups,momentum=.9,weight_decay=.0005)
    order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+20000))
    order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest();teacher=linear(ref['healthy_fit'],ref['heads']['coarse'])
    model.train();freeze_bn(model);trace=[]
    for j in range(0,256 if a.smoke else len(order),128):
        local=order[j:j+128];idx=fit[local]
        if a.mode=='F':h=ref['healthy_fit'][local]
        else:h=model(preprocess(images[idx]))
        logits=primary(h);labeled=mask[local]
        primary_loss=F.cross_entropy(logits[labeled],fine[idx][labeled]) if labeled.any() else logits.sum()*0
        if a.mode=='P':penalty=F.mse_loss(extra(h),teacher[local])
        elif a.mode=='B':penalty=F.mse_loss(extra(h),ref['healthy_fit'][local])
        else:penalty=h.sum()*0
        loss=primary_loss+a.lam*penalty
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite vision consumer continuation')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if j%4096==0:
            row=dict(batch=j//128,primary=float(primary_loss.detach()),alignment=float(penalty.detach()),elapsed_s=time.time()-start)
            trace.append(row);print(json.dumps(row),flush=True)
    assert paired.state_hash(bn_state(model))==bn_sha
    model.eval();hval=features(model,images,val) if a.mode!='F' else ref['healthy_validation']
    hfit=features(model,images,fit) if a.mode!='F' else ref['healthy_fit']
    healthy=linear(ref['healthy_validation'],ref['heads']['coarse'])
    readers={'fixed':ref['heads']['coarse']}
    if a.mode=='P':readers['protected']=copy.deepcopy(extra.state_dict())
    if a.mode=='B':
        readers['feature_map']=dict(weight=ref['heads']['coarse']['weight']@extra.weight.detach(),bias=ref['heads']['coarse']['bias'].clone())
    savedfits={}
    if a.mode in ['U','P']:
        ols=fit_legacy_heads(hfit,teacher)['ols'];readers['ols']={k:ols[k].float() for k in ['weight','bias']};savedfits['ols64']=ols
    if a.mode=='U':
        clf,mu,sd=fit_head(hfit[mask].double(),coarse[fit][mask]);readers['supervised_refit']=affine(clf,mu,sd)
    out=header(a,seed);out.update(mode=a.mode,lam=a.lam,smoke=a.smoke,trace=trace,
        configuration=dict(epochs=1,batch=128,dtype='float32',encoder_lr=.001,heads_lr=.01,
            momentum=.9,weight_decay=.0005,primary_labeled_fraction=.2,bn_buffers_frozen=True),
        source_reference_sha256=hashlib.sha256(a.cache.read_bytes()).hexdigest(),
        initial_encoder_sha256=paired.state_hash(ref['encoder']),final_encoder_sha256=paired.state_hash(model.state_dict()),
        minibatch_order_sha256=order_hash,bn_buffers_unchanged=True,
        nonlinear_layers_changed={prefix:paired.state_hash({k:v for k,v in model.state_dict().items() if k.startswith(prefix)})!=
            paired.state_hash({k:v for k,v in ref['encoder'].items() if k.startswith(prefix)}) for prefix in ['conv1.','layer1.','layer4.']},
        fine=paired.metrics(primary(hval),fine[val]),branches=[])
    for name,reader in readers.items():
        z=linear(hval,reader)
        out['branches'].append(dict(mode=name,reader_sha256=paired.state_hash(reader),
            contracts=evaluate_contracts(z,healthy,coarse[val],ref['thresholds'])))
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    torch.save(dict(encoder=model.state_dict(),primary=primary.state_dict(),readers=readers,
        extra=None if extra is None else extra.state_dict(),validation_features=hval,
        fitting_features=hfit if savedfits else None,**savedfits),folder/'endpoint.pt')
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','branch'],required=True)
    p.add_argument('--seed',type=int);p.add_argument('--cache',type=Path);p.add_argument('--mode',choices=['F','U','P','B'])
    p.add_argument('--lam',type=float,default=0.);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    (prepare if a.stage=='prepare' else branch)(a)
