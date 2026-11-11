"""Close the fixed failed visual distillation pilot without opening test."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'repro/results/CPUCONT'

def main():
    paths=sorted((BASE/'vision_protected_head_pilot').glob('h???.json'));assert len(paths)==6
    rows=[json.loads(p.read_text()) for p in paths];assert all(r['complete'] and r['phase']=='development' and r['seed']==3800 for r in rows)
    assert {(r['arch'],r['weight']) for r in rows}=={(a,w) for a in ['resnet18','vit_b32'] for w in [1,10,100]}
    assert all(not r['splits']['official_test_loaded'] and not r['branches'][0]['passes'] for r in rows)
    initial=sorted((BASE/'vision_head_initialization').glob('i???.json'));assert len(initial)==2
    for p in initial:
        r=json.loads(p.read_text());assert r['complete']
        assert all(abs(b['initial_kl'])<1e-5 and b['class_disagreement']==0 for b in r['branches'])
        assert all(v['disagreement']==0 for v in r['branches'][1]['contracts'].values())
    choice=dict(complete=True,plan_sha256=hashlib.sha256((ROOT/'docs/plans/2026-11-10-vision-protected-head-control.md').read_bytes()).hexdigest(),
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        followup_after_unconstrained_test_opened=True,development_only=True,test_confirmation_runs=0,
        backbones={a:dict(selected=None,reason='No fixed pilot weight passes both utility and all three consumers.') for a in ['resnet18','vit_b32']},
        evidence_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+initial})
    (BASE/'vision_protected_head_choice.json').write_text(json.dumps(choice,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['\\begin{table}[t]','\\centering','\\caption{Fixed one-forward distillation follow-up, development seed 3800 only. A learned coarse head is initialized to the healthy affine function. Classification and selective columns report disagreement in percent; fine accuracy is validation percent after the same final refit. None meets the fixed joint utility/consumer gate; no test evaluation follows.}','\\label{tab:vision_protected_head}','\\begin{tabular}{lrrrrr}','\\toprule','Backbone & $\\lambda$ & Fine accuracy & Class & Selective 80 & Selective 50 \\\\','\\midrule']
    for r in rows:
        b=r['branches'][0];v=b['validation'];c=v['contracts'];label='ResNet-18' if r['arch']=='resnet18' else 'ViT-B/32'
        lines.append(f"{label} & {r['weight']:g} & {100*v['own']['acc']:.2f} & {100*c['classification']['disagreement']:.2f} & {100*c['coverage80']['disagreement']:.2f} & {100*c['coverage50']['disagreement']:.2f} \\\\")
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (ROOT/'paper/vision_protected_head_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(pilot_runs=6,selected={a:None for a in choice['backbones']},initialization_checks=2,test_runs=0)))

if __name__=='__main__':main()
