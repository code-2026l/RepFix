"""Local manuscript tables from complete fixed-protocol numerical records."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];B=R/'repro/results/CPUCONT';P=R/'paper'


def main():
    routing=json.loads((B/'conditional_reference_choice.json').read_text())
    lines=[]
    for arch,title in [('resnet18','ResNet-18'),('vit_b32','ViT-B/32')]:
        value=routing['backbones'][arch];passed=[c for c in value['candidates'] if c['consumer_gate']]
        confidence=min(c['mean_referral'] for c in passed if c['gate']=='confidence')
        lines.append(f"{title} & {100*value['affine_oracle_mean_referral_at_one_percent']:.2f} & {100*confidence:.2f} & {100*value['lowest_referral_meeting_consumer_gate']:.2f} "+chr(92)*2)
    (P/'conditional_reference_table.tex').write_text(r'''\begin{table}[H]
\centering\small
\caption{Development reference-referral fractions (\%). Every tested gate shown
meets the one-percent class/selective disagreement limit on all three seeds,
but exceeds the fixed $50\%$ referral limit. The fractional oracle fixes the
affine reader; the tested columns allow both affine and residual readers.
It assumes unit fallback cost and is not a latency bound.}
\label{tab:conditional_reference}
\begin{tabular}{lrrr}\toprule
Backbone & Affine oracle & Confidence gate & Best tested gate \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule\end{tabular}
\end{table}
''',encoding='utf-8',newline='\n')
    pilot=json.loads((B/'vision_auxiliary_first_pilot_choice.json').read_text());assert pilot['complete']
    lines=[]
    for arch,title in [('resnet18','ResNet-18'),('vit_b32','ViT-B/32')]:
        for v in pilot['backbones'][arch]['candidates']:
            lines.append(f"{title} & ${v['alpha']:g}$ & ${v['gain_pp']:+.2f}$ & ${v['dose_matched_mse_ratio']:.3f}$ & No "+chr(92)*2)
    (P/'vision_auxiliary_pilot_table.tex').write_text(r'''\begin{table}[H]
\centering\small
\caption{Real reconstruction auxiliary on the independently fitted development
reference, one continuation seed per backbone. Gains compare the final refitted
primary reader with CE-only detachment D; MSE ratios use its dose-matched
detached decoder. Extension requires gain $\ge0.20$ points, ratio $\le0.95$ and
unchanged old decisions. No coefficient passes.}
\label{tab:vision_auxiliary_pilot}
\begin{tabular}{lrrrr}\toprule
Backbone & Auxiliary weight & Primary gain (points) & MSE ratio & Extend \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule\end{tabular}
\end{table}
''',encoding='utf-8',newline='\n')
    summary=B/'vision_reference_summary.json'
    if not summary.exists():return
    value=json.loads(summary.read_text());assert value['complete']
    lines=[]
    for group in value['reference_groups']:
        scores=group['accuracies'];cols=' & '.join(f"${100*np.mean(scores[k]):.2f}$" for k in ['F','N','H','D'])
        lines.append(f"{group['reference_seed']} & {group['label_count']} & {cols} & ${group['mean_gain_over_H_pp']:+.2f}$ "+chr(92)*2)
    (P/'reference_replication_table.tex').write_text(r'''\begin{table}[H]
\centering\small
\caption{ViT-B/32 transfer confirmation across four fitted references and
train/validation partitions. Each entry averages three continuation seeds.
F is an affine frozen-feature reader, N a learned nonlinear reader, H the
richer frozen-feature reader, and D the CE-only learned suffix. All use the
same fitting labels and final reader procedure. Accuracies and differences
are in percentage points. The ImageNet pretrained weights remain shared.}
\label{tab:reference_replication}
\begin{tabular}{lrrrrrr}\toprule
Reference & Labels & F & N & H & D & D$-$H \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule\end{tabular}
\end{table}
''',encoding='utf-8',newline='\n')
    numbers=[]
    for name,key in [('ReferenceViTGainH','H'),('ReferenceViTGainStrongest','strongest_frozen')]:
        stat=value['reference_level_statistics'][key]
        for suffix,number in [('',stat['mean']),('Low',stat['ci95'][0]),('High',stat['ci95'][1])]:
            numbers.append(f"\\newcommand{{\\{name+suffix}}}{{{100*number:.2f}}}")
    (P/'reference_replication_numbers.tex').write_text('\n'.join(numbers)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
