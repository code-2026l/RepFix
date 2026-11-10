"""Local, test-blind evaluation of the fixed nonlinear-suffix development gate."""
import hashlib,json
from pathlib import Path

def eligible(record,mode):
    assert record['complete'] and record['phase']=='development'
    assert not record['splits']['official_test_loaded'] and not record['splits']['test_metrics_computed']
    values={b['mode']:b['validation'] for b in record['branches']}
    own=values[mode]['own']['acc'];tol=1e-12
    return (own>=record['healthy']['acc']+.02-tol and own>=values['F']['own']['acc']+.02-tol
        and all(own>=values[b]['own']['acc']-.005-tol for b in ['U','N','H'])
        and all(v['disagreement']<=.01+tol for v in values[mode]['contracts'].values()))

def choice(paths):
    groups={};evidence={}
    for path in paths:
        path=Path(path);r=json.loads(path.read_text());groups.setdefault(r['arch'],{})[r['seed']]=r
        key=path.parent.name+'/'+path.name
        assert key not in evidence,key
        evidence[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    backbones={}
    for arch,records in groups.items():
        assert set(records)=={3800,3801,3802},(arch,set(records))
        present={b['mode'] for b in next(iter(records.values()))['branches']}
        candidates=[m for m in ['R64','U'] if m in present and all(eligible(r,m) for r in records.values())]
        # Both use a merged dense current suffix and the same retained old suffix.
        # Serving MACs therefore tie; prefer the smaller update rank as registered.
        backbones[arch]=dict(selected=next(iter(candidates),None),eligible=candidates,
            seeds=sorted(records),scope='one fixed reference, split and label mask',
            accuracies={str(s):{b['mode']:b['validation']['own']['acc'] for b in r['branches']} for s,r in records.items()})
    return dict(backbones=backbones,evidence_sha256=evidence,test_used_for_selection=False)
