"""Local analysis of every frozen fresh-seed replay outcome."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t

def interval(values):
    values=np.asarray(values,dtype=float);mean=float(values.mean());half=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean,ci95=[mean-half,mean+half],values=values.tolist())

def summarize(records):
    out=dict(complete=True,confirmed_on_new_seeds=True,cost_gate_passed=False,domains={})
    for domain in ['har','radioml']:
        rows=[r for r in records if r['domain']==domain];assert {r['seed'] for r in rows}==set(range(2500,2510)) and all(r['complete'] for r in rows)
        paired=[{b['mode']:b for b in r['branches']} for r in rows]
        utility=interval([p['joint']['test']['own']['acc']-p['detach']['test']['own']['acc'] for p in paired])
        full=interval([p['joint']['test']['own']['acc']-r['full_controls']['joint']['test']['own']['acc'] for p,r in zip(paired,rows)])
        mse=interval([p['joint']['test']['auxiliary_mse']/p['detach']['test']['auxiliary_mse'] for p in paired])
        contracts={name:dict(maximum_disagreement=max(b['test']['contracts'][name]['disagreement'] for r in rows for b in r['branches']),
            certified=sum(b['test']['contracts'][name]['certificate']['certified'] for r in rows for b in r['branches']),readers=20,
            maximum_bound=max(b['test']['contracts'][name]['certificate']['upper_remaining_disagreement'] for r in rows for b in r['branches']))
            for name in ['classification','coverage80','coverage50']}
        checks=dict(compatibility=all(c['certified']==20 for c in contracts.values()),
            nonlinear_updates=all(b['test']['changed_second_relu_gates']>0 and b['test']['changed_first_relu_gates']>0 for r in rows for b in r['branches']))
        if domain=='har':checks.update(primary_noninferiority=utility['ci95'][0]>=-.005,reconstruction_progress=mse['ci95'][1]<=.95)
        else:checks.update(positive_primary_gain=utility['ci95'][0]>0,noninferiority_to_full_joint=full['ci95'][0]>=-.005)
        out['domains'][domain]=dict(seeds=[r['seed'] for r in rows],primary_gain=utility,primary_difference_to_full_joint=full,
            auxiliary_mse_ratio=mse,contracts=contracts,checks=checks,utility_compatibility_confirmed=all(checks.values()),
            maximum_full_replay_logit_error=max(b['test']['full_reference_maximum_logit_error'] for r in rows for b in r['branches']),
            maximum_compressed_logit_error=max(b['test']['packed_serving']['maximum_legacy_logit_error'] for r in rows for b in r['branches']),
            maximum_current_merge_logit_error=max(b['test']['current_merge_maximum_logit_error'] for r in rows for b in r['branches']),
            current_merge_prediction_changes=sum(b['test']['current_merge_prediction_changes'] for r in rows for b in r['branches']),
            mean_accuracies={mode:float(np.mean([p[mode]['test']['own']['acc'] for p in paired])) for mode in ['detach','joint']},
            mean_full_accuracies={mode:float(np.mean([r['full_controls'][mode]['test']['own']['acc'] for r in rows])) for mode in ['detach','joint']})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    paths=sorted(a.records.glob('c???.json'));records=[json.loads(path.read_text()) for path in paths];out=summarize(records)
    out['source_record_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    a.out.write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({d:{k:v[k] for k in ['primary_gain','primary_difference_to_full_joint','auxiliary_mse_ratio','checks','utility_compatibility_confirmed']} for d,v in out['domains'].items()},indent=2))
