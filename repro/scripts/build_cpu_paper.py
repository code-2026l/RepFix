"""Compile and inspect the revised paper locally, with compact diagnostics."""
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper'
QA=ROOT/'tmp/cpu_build'
QA.mkdir(parents=True,exist_ok=True)
source=(PAPER/'RepFix_ICLR2027.tex').read_text(encoding='utf-8')
baseline=ROOT.parent/'_remote_wt/paper'
copied=[]
for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',source):
    target=(PAPER/name).resolve()
    origin=(baseline/name).resolve()
    if not target.is_relative_to(PAPER.resolve()) or not origin.is_relative_to(baseline.resolve()):
        raise ValueError('Unexpected figure path')
    if not target.exists():
        if not origin.exists():
            raise FileNotFoundError(name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(origin,target)
        copied.append(name)
commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','RepFix_ICLR2027.tex'],
          ['bibtex','RepFix_ICLR2027'],
          ['pdflatex','-interaction=nonstopmode','-halt-on-error','RepFix_ICLR2027.tex'],
          ['pdflatex','-interaction=nonstopmode','-halt-on-error','RepFix_ICLR2027.tex']]
for i,command in enumerate(commands):
    result=subprocess.run(command,cwd=PAPER,capture_output=True)
    (QA/('pass%d.txt'%i)).write_bytes(result.stdout+b'\n'+result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout.decode('utf-8',errors='replace')[-3000:])
log=(PAPER/'RepFix_ICLR2027.log').read_text(encoding='utf-8',errors='replace')
warnings=[line for line in log.splitlines() if 'undefined' in line.lower() or 'Overfull' in line or 'multiply defined' in line]
aux=(PAPER/'RepFix_ICLR2027.aux').read_text(encoding='utf-8',errors='replace')
labels={m.group(1):dict(number=m.group(2),page=int(m.group(3)))
        for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}',aux)}
result=dict(copied_baseline_figures=copied,warnings=warnings,
            labels={k:labels[k] for k in ['sec:conclusion','app:boundary_proof','fig:paired_dose','app:cpu_paired','tab:cpu_paired','tab:cpu_adam'] if k in labels},
            output=str(PAPER/'RepFix_ICLR2027.pdf'))
(QA/'build_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
