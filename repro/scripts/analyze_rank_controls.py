"""Describe RS1 crossings without turning partial contraction into collapse."""
import argparse
import json
from pathlib import Path
import numpy as np
from synth_beta_joint import cross, value_at


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',default='repro/results/RS1')
    ap.add_argument('--out',default='results/OPERATOR_AUDIT/rs1_summary.json')
    a=ap.parse_args();rows=[]
    for path in sorted(Path(a.directory).glob('*.json')):
        d=json.loads(path.read_text());pts=sorted((r['alpha'],r['sigma_h']) for r in d['records'])
        for anchor in (0.,.0003):
            healthy=value_at(pts,anchor)
            if healthy is None or healthy<=0:continue
            tail=[(x,y) for x,y in pts if x>=anchor]
            rows.append(dict(file=path.name,m=d['m'],per_head_rank=d['r'],
                combined_rank=min(d['m'],2*d['r']),anchor=anchor,
                half_height_crossing=cross(tail,healthy/2),
                minimum_relative_scatter=min(y/healthy for _,y in tail),
                endpoint_relative_scatter=tail[-1][1]/healthy,
                floor_below_one_percent=min(y/healthy for _,y in tail)<.01))
    fits=[]
    for rank in sorted({x['per_head_rank'] for x in rows}):
        for anchor in (0.,.0003):
            subset=[x for x in rows if x['per_head_rank']==rank and x['anchor']==anchor and x['half_height_crossing']]
            if len(subset)>=3:
                x=np.log([r['m'] for r in subset]);y=np.log([r['half_height_crossing'] for r in subset])
                slope,intercept=np.polyfit(x,y,1)
                fits.append(dict(rank=rank,anchor=anchor,n=len(subset),descriptive_slope=float(slope),
                    all_resolve_floor=all(r['floor_below_one_percent'] for r in subset)))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows,fits=fits,
        warning='Aggregate curves only; no independent-seed confidence intervals. Half-height is not a verified collapse threshold.'),indent=2))
    for row in rows:
        if row['anchor']==.0003:print(row)
    print('FITS',fits)


if __name__=='__main__':main()
