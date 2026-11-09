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
            labels={k:labels[k] for k in ['sec:conclusion','app:boundary_proof','fig:overview','fig:paired_dose',
                'fig:nullspace_control','fig:benefit_confirmation','tab:benefit_confirmation',
                'app:benefit_confirmation','prop:consumer_nullspace','tab:benefit_contrasts',
                'app:compiled_interface','cor:compiled_interface','fig:compiled_control',
                'fig:compiled_frontier','tab:compiled_full','tab:compiled_contrasts',
                'app:nonlinear_interface','fig:nonlinear_interface','fig:nonlinear_validation',
                'tab:nonlinear_development',
                'fig:scale_law','app:cpu_paired','tab:cpu_paired','tab:cpu_adam',
                'app:head_state','tab:head_state','app:consumer_audit','tab:consumer_audit','tab:cnn_continuation'] if k in labels},
            output=str(PAPER/'RepFix_ICLR2027.pdf'))
extraction=subprocess.run(['pdftotext','-layout',str(PAPER/'RepFix_ICLR2027.pdf'),'-'],
                          capture_output=True,check=True)
pages=[p for p in extraction.stdout.decode('utf-8',errors='replace').split('\f') if p.strip()]
def heading_pages(title):
    return [i+1 for i,p in enumerate(pages) if any(
        re.sub('[^A-Z]','',line.upper())==title for line in p.splitlines())]
statements=heading_pages('AIUSESTATEMENT')
references=heading_pages('REFERENCES')
if len(statements)!=1 or not references:
    raise RuntimeError('Cannot identify the main-text boundary or reference heading')
result.update(total_pages=len(pages),main_text_pages=statements[0]-1,
              references_start_page=references[0])
if result['main_text_pages']>9:
    raise RuntimeError('Main text exceeds nine pages, including continuation text and floats')
(QA/'build_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
