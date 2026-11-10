"""Local selection from all fixed development routing candidates."""
from pathlib import Path
import argparse,hashlib,json

def main(root,out):
    paths=sorted(root.glob('p???.json'));rows=[json.loads(p.read_text()) for p in paths]
    assert len(rows)==6 and all(r['complete'] and not r['splits']['official_test_loaded'] for r in rows)
    result=dict(complete=True,phase='development',test_confirmation_runs=0,backbones={},
        record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    for arch in ['resnet18','vit_b32']:
        own=[r for r in rows if r['arch']==arch];assert {r['seed'] for r in own}==set(range(3800,3803))
        candidates=[]
        for b in own[0]['branches']:
            key=(b['reader'],b['gate'],b['target_referral'])
            matches=[next(v for v in r['branches'] if (v['reader'],v['gate'],v['target_referral'])==key) for r in own]
            mean=sum(v['referral'] for v in matches)/3;gate=all(v['passes_consumer_gate'] for v in matches)
            candidates.append(dict(reader=key[0],gate=key[1],target_referral=key[2],mean_referral=mean,
                max_disagreement=max(v['maximum_disagreement'] for v in matches),consumer_gate=gate,passes=gate and mean<=.5))
        passing=[v for v in candidates if v['passes']]
        ranks=['confidence','boundary','normalized_boundary','learned_disagreement']
        passing.sort(key=lambda v:(v['mean_referral'],v['reader']!='affine',ranks.index(v['gate']),v['target_referral']))
        certified=[v for v in candidates if v['consumer_gate']]
        result['backbones'][arch]=dict(selected=passing[0] if passing else None,candidates=candidates,
            lowest_referral_meeting_consumer_gate=min(v['mean_referral'] for v in certified),
            affine_oracle_mean_referral_at_one_percent=sum(r['readers']['affine']['oracle']['0.01']['minimum_fraction'] for r in own)/3)
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:{n:v[n] for n in ['selected','lowest_referral_meeting_consumer_gate','affine_oracle_mean_referral_at_one_percent']} for k,v in result['backbones'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.records,a.out)
