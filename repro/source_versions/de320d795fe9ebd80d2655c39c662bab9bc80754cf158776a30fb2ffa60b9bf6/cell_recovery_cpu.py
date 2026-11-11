"""Fixed-feature consumer-cell/capacity development audit, CPU numerical work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_cells import cell_spec,cell_loss,cell_logodds_loss,slice_spec
from consumer_contracts import confidence_thresholds,evaluate_contracts
from legacy_head_fit import fit_legacy_heads
from recovery_readers import RecoveryReader,export_reader,pack_outputs,serve_outputs,reader_logits

ARCHITECTURES=['affine','quadratic','residual32']
OBJECTIVES=['kl1','kl2','kl4','centered_mse','cell_kl','cell_logodds']
SOURCES=['cell_recovery_cpu.py','consumer_cells.py','recovery_readers.py','consumer_contracts.py',
         'legacy_head_fit.py','benefit_compatibility_cpu.py','paired_readout_continuation.py']


def reader_loss(z,teacher,spec,objective):
    if objective=='cell_kl':return cell_loss(z,spec)
    if objective=='cell_logodds':return cell_logodds_loss(z,spec)
    if objective=='centered_mse':
        return F.mse_loss(z.double()-z.double().mean(1,keepdim=True),teacher-teacher.mean(1,keepdim=True))
    temperature=float(objective[2:]);logq=(z.double()/temperature).log_softmax(1)
    return F.kl_div(logq,(teacher/temperature).softmax(1),reduction='batchmean')*temperature**2


@torch.no_grad()
def score_reader(model,features,part,healthy_logits,thresholds,reader):
    primary,z=serve_outputs(features,pack_outputs(model.primary,reader));y=part[1]
    return dict(own=paired.metrics(primary,y),contracts=evaluate_contracts(z.double(),healthy_logits,y,thresholds),
        standalone_serving_prediction_changes=int((z.argmax(1)!=reader_logits(features,reader).argmax(1)).sum()))


def fit_reader(h,teacher,thresholds,initial,architecture,objective,seed,epochs):
    torch.manual_seed(seed+42000)
    mu,sd=h.mean(0,keepdim=True),h.std(0,keepdim=True).clamp_min(1e-6)
    reader=RecoveryReader(h.shape[1],teacher.shape[1],architecture,mu,sd,initial)
    opt=torch.optim.Adam(reader.parameters(),lr=.001,weight_decay=.0001)
    spec=cell_spec(teacher,list(thresholds.values()));rng=torch.Generator().manual_seed(seed+41000)
    order_hash=hashlib.sha256();trace=[];start=time.time()
    for epoch in range(epochs):
        order=torch.randperm(len(h),generator=rng);order_hash.update(order.numpy().tobytes())
        for j in range(0,len(order),1024):
            idx=order[j:j+1024];z=reader(h[idx]);loss=reader_loss(z,teacher[idx],slice_spec(spec,idx),objective)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reader recovery objective')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(reader.parameters(),5.);opt.step()
        if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
            trace.append(dict(epoch=epoch+1,loss=float(loss.detach())))
    reader.eval();native=export_reader(reader)
    return reader,native,dict(architecture=architecture,objective=objective,epochs=epochs,trace=trace,
        elapsed_s=time.time()-start,minibatch_order_sha256=order_hash.hexdigest(),
        serving_reader_sha256=paired.state_hash(native),parameters=sum(p.numel() for p in reader.parameters()))


def run(a):
    start=time.time();record=json.loads(a.record.read_text());assert record['phase']=='development' and record['complete']
    domain,seed=record['domain'],record['seed'];torch.manual_seed(seed);np.random.seed(seed)
    cfg=SimpleNamespace(domain=domain,seed=seed,phase='development',limit_fit=0,limit_eval=0)
    model,data,kind,splits=bc.load(cfg);assert splits==record['splits'] and 'test' not in data
    folder=a.record.parent/(a.record.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ref['model']);model.eval()
    A=ref['head']['weight'].double()/ref['sd'].reshape(-1);B=ref['head']['bias'].double()-A@ref['mean'].reshape(-1)
    healthy_fit=paired.embed(model,data['x'],batch=1024);healthy_val=paired.embed(model,data['validation'][0],batch=1024)
    target=healthy_fit@A.T+B;oldval=healthy_val@A.T+B;thresholds=confidence_thresholds(target)
    donor_path=folder/'joint_0.pt';donor_record=a.record
    if domain=='har':
        assert a.alignment_record
        donor_record=a.alignment_record;aligned=json.loads(donor_record.read_text());assert aligned['domain']=='har' and aligned['seed']==seed
        donor_path=donor_record.parent/(donor_record.stem+'_states')/'feature_map_1.pt'
    donor=torch.load(donor_path,map_location='cpu',weights_only=True);model.load_state_dict(donor['model']);model.eval()
    model_sha=paired.state_hash(model.state_dict());h=paired.embed(model,data['x'],batch=1024);val=paired.embed(model,data['validation'][0],batch=1024)
    if a.smoke:h=h[:512];target=target[:512];val=val[:512];oldval=oldval[:512];data['validation']=tuple(x[:512] for x in data['validation'])
    initial=fit_legacy_heads(h,target)['ols'];native=dict(kind='affine',**{k:initial[k].float() for k in ['weight','bias']})
    own=paired.metrics(F.linear(val.float(),model.primary.weight,model.primary.bias),data['validation'][1])
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='development',route='fixed_features',domain=domain,seed=seed,
        smoke=a.smoke,plan_sha256=a.plan_sha256,metric_addendum_plan_sha256=a.addendum_sha256,
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES},
        source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),donor_record_sha256=hashlib.sha256(donor_record.read_bytes()).hexdigest(),
        reference_model_sha256=paired.state_hash(ref['model']),donor_model_sha256=model_sha,
        splits=splits,thresholds=thresholds,controls={r['mode']:r['validation'] for r in record['branches'] if r['mode'] in ['detach','joint']},
        donor=dict(own=own,auxiliary_mse=float(paired.auxiliary_loss(model,val.float(),data['validation'][2],kind))),
        configuration=dict(architectures=ARCHITECTURES,objectives=OBJECTIVES,epochs=60,batch=1024,lr=.001,
            gradient_clip=5.,weight_decay=.0001,encoder_updates=False,labels_used_for_reader_fitting=False,reader_runtime_dtype='float32'),branches=[])
    destination=a.out.parent/(a.out.stem+'_states');destination.mkdir(exist_ok=True,parents=True)
    torch.save(dict(model=donor['model'],reference=ref['model'],A=A,B=B,train_features=h,validation_features=val,
        teacher_logits=target,healthy_validation_logits=oldval,validation_labels=data['validation'][1],
        thresholds=thresholds,ols64=initial),destination/'features.pt')
    baseline=dict(architecture='affine',objective='ols',serving_reader_sha256=paired.state_hash(native))
    baseline.update(score_reader(model,val,data['validation'],oldval,thresholds,native));out['branches'].append(baseline)
    torch.save(dict(reader=native),destination/'affine_ols.pt')
    for architecture in ARCHITECTURES:
        for objective in (['cell_kl','cell_logodds'] if a.smoke else OBJECTIVES):
            reader,native,row=fit_reader(h,target,thresholds,initial,architecture,objective,seed,1 if a.smoke else 60)
            row.update(score_reader(model,val,data['validation'],oldval,thresholds,native))
            with torch.no_grad():
                z=reader(val);folded=reader_logits(val,native)
                row['folding_precision']=dict(maximum_logit_error=float((z-folded).abs().max()),
                    prediction_changes=int((z.argmax(1)!=folded.argmax(1)).sum()))
            name=architecture+'_'+objective+'.pt';row['state_file']=name
            torch.save(dict(reader=native,unfolded=reader.state_dict()),destination/name)
            out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
            print(json.dumps(dict(seed=seed,domain=domain,architecture=architecture,objective=objective,
                changes={c:x['disagreement'] for c,x in row['contracts'].items()},elapsed_s=time.time()-start)),flush=True)
    assert paired.state_hash(model.state_dict())==model_sha
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--alignment-record',type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--addendum-sha256',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
