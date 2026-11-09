"""Predeclared CPU development/confirmation of a compatible nonlinear channel."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

import benefit_compatibility_cpu as previous
import paired_readout_continuation as paired
from consumer_risk import certificate
from nonlinear_interface import NativeInterface, NonlinearSideInterface, continued_adam, encoder_accounting
from readout_nullspace import readout_projector
from readout_transfer_audit import fit_head


def load(args):
    base, data, kind, splits = previous.load(args)
    fraction = args.labels or (.01 if args.domain == 'har' else .2)
    mask = torch.zeros(len(data['x']), dtype=torch.bool)
    order = torch.randperm(len(mask), generator=torch.Generator().manual_seed(args.seed+501))
    mask[order[:max(1, round(len(mask)*fraction))]] = True
    if len(torch.unique(data['y'][mask])) != base.primary.out_features:
        raise ValueError('The fixed labeled fitting subset omits a primary class')
    data['labeled'] = mask
    data['x'] = data['x'].double()
    for name in ['validation', 'test']:
        if name in data:
            x, y, a = data[name]
            data[name] = (x.double(), y, a)
    splits.update(primary_label_fraction=fraction, labeled_fitting_examples=int(mask.sum()),
                  labeled_mask_sha256=paired.state_hash(mask))
    return NativeInterface(base), data, kind, splits


def common_model_state(model):
    if not isinstance(model, NonlinearSideInterface):
        return {key:value for key,value in model.state_dict().items()
                if not key.startswith(('anchor_', 'projector'))}
    result = {}
    for key, value in model.state_dict().items():
        if key.startswith(('base_trunk.', 'base_final.', 'primary.', 'aux.')):
            result[key.replace('base_trunk.', 'trunk.').replace('base_final.', 'final.')] = value
    return result


class AnchoredInterface(nn.Module):
    def __init__(self, current, reference, projector):
        super().__init__()
        self.current = current
        self.reference = reference
        self.register_buffer('projector', projector)

    @property
    def primary(self):
        return self.current.primary

    @property
    def aux(self):
        return self.current.aux

    def forward(self, x):
        base = self.reference(x)
        return base+(self.current(x)-base) @ self.projector


@torch.no_grad()
def score(model, part, reference, head, mu, sd, kind, monitor=None):
    x, y, auxiliary = part
    h = paired.embed(model, x)
    z, z0 = head((h-mu)/sd), head((reference-mu)/sd)
    hc, rc = h-h.mean(0), reference-reference.mean(0)
    result = dict(own=paired.metrics(previous.current_logits(model,h),y), frozen=paired.metrics(z,y),
        reference=paired.metrics(z0,y), prediction_changes=int((z.argmax(1)!=z0.argmax(1)).sum()),
        disagreement=float((z.argmax(1)!=z0.argmax(1)).double().mean()),
        max_absolute_healthy_logit_error=float((z-z0).abs().max()),
        relative_trace=float(hc.square().sum()/rc.square().sum().clamp_min(1e-30)),
        auxiliary_mse=float(paired.auxiliary_loss(model,h.float(),auxiliary,kind)))
    if monitor is not None:
        bound = certificate(z.argmax(1).numpy(),z0.argmax(1).numpy(),monitor,.05/200)
        remainder = np.setdiff1d(np.arange(len(y)),monitor)
        bound.update(budget=.05,certified=bound['upper_remaining_disagreement']<=.05,
            remainder_loss_increase=float((z0.argmax(1)[remainder]==y[remainder]).double().mean() -
                                         (z.argmax(1)[remainder]==y[remainder]).double().mean()),
            remainder_disagreement=float((z.argmax(1)[remainder]!=z0.argmax(1)[remainder]).double().mean()))
        result['consumer_audit'] = bound
    return result


@torch.no_grad()
def diagnose(model, x, base, head, mu, sd):
    h = paired.embed(model,x)
    z,z0 = head((h-mu)/sd),head((base-mu)/sd)
    changes = int((z.argmax(1)!=z0.argmax(1)).sum())
    return dict(anchor_examples=len(x),prediction_changes=changes,disagreement=changes/len(x),
        max_absolute_logit_change=float((z-z0).abs().max()),threshold=.05,triggered=changes/len(x)>.05)


def specs(args):
    widths = [int(value) for value in args.widths.split(',')]
    doses = [float(value) for value in args.doses.split(',')]
    detached_dose = doses[0] if len(doses)==1 else 1.
    result = [('D',0,detached_dose)]+[('J',0,dose) for dose in doses]
    if args.phase=='confirmation':
        assert len(widths)==len(doses)==1
        result += [(mode,0,doses[0]) for mode in ['CD','CJ','GD','GJ']]
    result += [(mode,width,dose) for dose in doses for width in widths for mode in ['UD','UJ','TD','TJ']]
    return result


def create_branch(healthy, moments, p, mode, width):
    if mode.startswith(('T','U')):
        projector = p if mode.startswith('T') else torch.eye(p.shape[0],dtype=torch.float64)
        model = NonlinearSideInterface(healthy,width,projector)
    else:
        model = copy.deepcopy(healthy)
    assert paired.state_hash(common_model_state(model)) == paired.state_hash(healthy.state_dict())
    optimizer = continued_adam(model,moments)
    assert paired.state_hash(optimizer.state_dict()['state']) == paired.state_hash(moments['state'])
    if mode.startswith('C'):
        model.constrain_last(p)
    return model,optimizer


def endpoint_model(model, healthy, p, mode):
    if mode.startswith('G'):
        return AnchoredInterface(model,healthy,p)
    if mode.startswith(('T','U')):
        return model.fuse()
    return model.fold()


@torch.no_grad()
def benchmark(models, inputs):
    timings = {}
    for size in [1,1024]:
        sample = inputs[:min(size,len(inputs))]
        operations = {name:(lambda model=model: previous.current_logits(model,model(sample)))
                      for name,model in models.items()}
        for operation in operations.values():
            for _ in range(5): operation()
        samples = {name:[] for name in operations}
        names = list(operations)
        for repeat in range(21):
            for name in names if repeat%2==0 else list(reversed(names)):
                start = time.perf_counter();operations[name]();samples[name].append(time.perf_counter()-start)
        timings[str(size)] = {name:dict(batch_size=len(sample),repeats=21,median_seconds=float(np.median(values)))
                             for name,values in samples.items()}
    fused = models['fused_side']
    calls = []
    handles = [layer.register_forward_hook(lambda *_: calls.append(1))
               for layer in [*fused.trunk.modules(),fused.final] if isinstance(layer,nn.Linear)]
    fused(inputs[:1])
    for handle in handles: handle.remove()
    assert len(calls)==3
    assert not any('base_' in key or 'side_' in key or 'projector' in key for key in fused.state_dict())
    return dict(timings=timings,deployed_affine_calls=3,
        ordinary_encoder=encoder_accounting(models['ordinary']),fused_encoder=encoder_accounting(fused),
        frozen_base_weights_retained_inside_fused_encoder=True,
        encoder_precision='float64',timing_path='encoder plus current head')


def run(args):
    start = time.time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model,data,kind,splits = load(args)
    x,y,a,mask = [data[key] for key in ['x','y','a','labeled']]
    warm = args.warm_epochs or (60 if args.domain=='har' else 25)
    cont = args.cont_epochs or (30 if args.domain=='har' else 25)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    rng = torch.Generator().manual_seed(args.seed+10000)
    for epoch in range(warm):
        order = torch.randperm(len(x),generator=rng)
        for offset in range(0,len(order),256):
            idx = order[offset:offset+256]
            previous.step(model,optimizer,x[idx],y[idx],a[idx],mask[idx],kind,1.,True)
        if epoch==0 or (epoch+1)%10==0:
            print(dict(stage='warm',epoch=epoch+1,elapsed_s=time.time()-start),flush=True)
    model.eval()
    healthy = copy.deepcopy(model)
    reference,moments = copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict())
    ref_sha,adam_sha = paired.state_hash(reference),paired.state_hash(moments)
    hfit = paired.embed(model,x)
    head,mu,sd = fit_head(hfit[mask],y[mask])
    p,projection_info = readout_projector(head.weight.detach(),sd)
    bases = {name:paired.embed(model,data[name][0]) for name in ['validation','test'] if name in data}
    states = args.out.parent/(args.out.stem+'_states');states.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=reference,optimizer=moments,head=head.state_dict(),mean=mu,sd=sd,projector=p,splits=splits),states/'reference.pt')
    source_names = ['nonlinear_interface_cpu.py','nonlinear_interface.py','compiled_readout.py',
                    'benefit_compatibility_cpu.py','paired_readout_continuation.py','readout_nullspace.py','consumer_risk.py']
    out = dict(protocol='predeclared nonlinear compatible channel with explicit dense deployment budget',
        phase=args.phase,seed=args.seed,domain=args.domain,auxiliary=kind,plan_sha256=args.plan_sha256,
        selection_sha256=args.selection_sha256,device='cpu',cuda_visibility='',torch_threads=1,
        torch_version=torch.__version__,python_version=platform.python_version(),splits=splits,
        configuration=dict(width=64,channel_widths=args.widths,doses=args.doses,batch=256,lr=.001,
            warm_epochs=warm,continuation_epochs=cont,encoder_dtype='float64',head_parameter_dtype='float32'),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names},
        reference_model_sha256=ref_sha,reference_optimizer_sha256=adam_sha,projector=projection_info,
        branches=[],complete=False)
    monitor = None
    if args.phase=='confirmation':
        n = len(data['test'][1]);monitor = np.random.default_rng(3890001+args.seed).choice(n,min(1024,n-1),replace=False)
        out['consumer_bank'] = dict(candidates=200,delta=.05,budget=.05,monitor_inputs=len(monitor),
            monitor_membership_sha256=hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest(),
            independent_of_training=True,separate_from_previous_banks=True)
    for mode,width,dose in specs(args):
        model,optimizer = create_branch(healthy,moments,p,mode,width)
        key = '%s_k%d_a%g'%(mode,width,dose)
        shared_sha = paired.state_hash(common_model_state(model))
        assert shared_sha==ref_sha
        initial_side_sha = paired.state_hash(dict(trunk=model.side_trunk.state_dict(),final=model.side_final.state_dict())) if width else None
        initial_model_sha = paired.state_hash(model.state_dict())
        anchor_n = min(512,len(x));anchor_x,anchor_h = x[:anchor_n],hfit[:anchor_n]
        def audited():
            return AnchoredInterface(model,healthy,p) if mode.startswith('G') else model
        row = dict(key=key,mode=mode,channel_width=width,alpha=dose,
            initial_shared_model_sha256=shared_sha,initial_model_sha256=initial_model_sha,
            initial_shared_optimizer_state_sha256=paired.state_hash(optimizer.state_dict()['state']),
            initial_side_sha256=initial_side_sha,
            diagnostic_trace=[dict(epoch=0,**diagnose(audited(),anchor_x,anchor_h,head,mu,sd))])
        rng = torch.Generator().manual_seed(args.seed+20000);order_sha = hashlib.sha256();begin = time.time()
        for epoch in range(cont):
            order = torch.randperm(len(x),generator=rng);order_sha.update(order.numpy().tobytes())
            for offset in range(0,len(order),256):
                idx = order[offset:offset+256]
                previous.step(model,optimizer,x[idx],y[idx],a[idx],mask[idx],kind,dose,mode.endswith('D'),
                    reference=hfit[idx] if mode.startswith('G') else None,
                    projector=p if mode.startswith('G') else None)
            if epoch==0 or (epoch+1)%5==0 or epoch+1==cont:
                row['diagnostic_trace'].append(dict(epoch=epoch+1,**diagnose(audited(),anchor_x,anchor_h,head,mu,sd)))
                print(dict(stage='continue',key=key,epoch=epoch+1,elapsed_s=time.time()-begin),flush=True)
        if mode.startswith(('T','U')):
            assert paired.state_hash(model.base_trunk.state_dict())==paired.state_hash(healthy.trunk.state_dict())
            assert paired.state_hash(model.base_final.state_dict())==paired.state_hash(healthy.final.state_dict())
        if mode.startswith('C'):
            assert paired.state_hash(model.trunk.state_dict())==paired.state_hash(healthy.trunk.state_dict())
        deployed = endpoint_model(model,healthy,p,mode)
        unfused = audited()
        fold_error = max(float((paired.embed(unfused,data[name][0])-paired.embed(deployed,data[name][0])).abs().max())
                         for name in ['validation','test'] if name in data)
        assert fold_error<1e-10,(key,fold_error)
        row.update(validation=score(deployed,data['validation'],bases['validation'],head,mu,sd,kind),
            fusion_max_absolute_feature_error=fold_error,minibatch_order_sha256=order_sha.hexdigest(),
            final_model_sha256=paired.state_hash(model.state_dict()),
            final_optimizer_sha256=paired.state_hash(optimizer.state_dict()),elapsed_s=time.time()-begin)
        if width:
            row['nonlinear_prefix_changed'] = not torch.equal(model.side_trunk[0].weight,
                                                             healthy.trunk[0].weight[:width])
        if args.phase=='confirmation':
            row['test'] = score(deployed,data['test'],bases['test'],head,mu,sd,kind,monitor)
        if mode.startswith(('T','G','C')):
            for name in ['validation','test']:
                if name not in row: continue
                assert row[name]['max_absolute_healthy_logit_error']<1e-8,(key,row[name])
                assert row[name]['prediction_changes']==0
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),states/(key+'.pt'))
        if mode.startswith(('T','U','C')):
            torch.save(dict(model=deployed.state_dict()),states/(key+'_deployed.pt'))
            row['deployed_model_sha256'] = paired.state_hash(deployed.state_dict())
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'),out)
    assert len({row['minibatch_order_sha256'] for row in out['branches']})==1
    if args.phase=='confirmation':
        models = {}
        for mode in ['D','CD','GJ','TJ']:
            row = next(row for row in out['branches'] if row['mode']==mode)
            model,_ = create_branch(healthy,moments,p,mode,row['channel_width'])
            model.load_state_dict(torch.load(states/(row['key']+'.pt'),weights_only=True)['model']);model.eval()
            models[mode] = model
        models = dict(ordinary=models['D'].fold(),frozen_affine=models['CD'].fold(),
            unfused_side=models['TJ'],fused_side=models['TJ'].fuse(),
            two_encoders=AnchoredInterface(models['GJ'],healthy,p))
        out['deployment'] = benchmark(models,data['validation'][0])
    out.update(complete=True,elapsed_s=time.time()-start)
    paired.save_json(args.out,out)
    print(dict(stage='complete',output=args.out.name,elapsed_s=out['elapsed_s']),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',choices=['har','radioml'],required=True)
    parser.add_argument('--phase',choices=['development','confirmation'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--widths',default='16,32')
    parser.add_argument('--doses')
    parser.add_argument('--labels',type=float,default=0.)
    parser.add_argument('--warm-epochs',type=int,default=0)
    parser.add_argument('--cont-epochs',type=int,default=0)
    parser.add_argument('--limit-fit',type=int,default=0)
    parser.add_argument('--limit-eval',type=int,default=0)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--selection-sha256')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.doses = args.doses or ('1,10' if args.domain=='har' else '1')
    if len(args.plan_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.plan_sha256):
        parser.error('The committed local plan SHA-256 is required')
    if args.phase=='confirmation' and not args.limit_fit and not args.selection_sha256:
        parser.error('Confirmation requires a frozen choice hash')
    run(args)


if __name__=='__main__':main()
