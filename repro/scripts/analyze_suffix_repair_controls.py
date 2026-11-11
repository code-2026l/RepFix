"""Local descriptive flows for direct old readouts in the fixed repair screen."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'repro/results/CPUCONT'

def main():
    paths=sorted((BASE/'suffix_repair_development').glob('a???.json'));assert len(paths)==2
    out=dict(complete=True,analysis='local',phase='development',source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},backbones={},
        evidence_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    lines=['\\begin{table}[t]','\\centering','\\caption{Direct old coarse readout on updated visual features: descriptive means over the three fixed development seeds. All quantities are percentages on the same 5000 validation inputs. Accuracy can improve while formerly correct predictions regress. Labels diagnose flows here; they do not fit or select a repair.}','\\label{tab:suffix_direct_flows}','\\begin{tabular}{lrrrrr}','\\toprule','Backbone & Healthy acc. & Direct acc. & Changed & Correct $\\to$ wrong & Wrong $\\to$ correct \\\\','\\midrule']
    for path in paths:
        r=json.loads(path.read_text());assert r['complete'] and len(r['branches'])==3
        rows=[b['repairs']['frozen']['contracts']['classification'] for b in r['branches']]
        fields=dict(healthy_accuracy=[x['reference_metrics']['acc'] for x in rows],direct_accuracy=[x['metrics']['acc'] for x in rows],
            changed=[x['disagreement'] for x in rows],negative=[x['flows']['negative_flip_rate'] for x in rows],
            positive=[x['flows']['wrong_to_correct']/x['flows']['examples'] for x in rows])
        values={k:dict(mean=statistics.mean(v),values=v) for k,v in fields.items()};out['backbones'][r['arch']]=values
        label='ResNet-18' if r['arch']=='resnet18' else 'ViT-B/32'
        lines.append(label+' & '+' & '.join(f"{100*values[k]['mean']:.2f}" for k in fields)+r' \\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (BASE/'suffix_repair_flow_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    (ROOT/'paper/suffix_direct_flows_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({a:{k:round(100*v['mean'],2) for k,v in values.items()} for a,values in out['backbones'].items()}))

if __name__=='__main__':main()
