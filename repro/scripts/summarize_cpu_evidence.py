"""Analyze retrieved CPU records and generate manuscript numbers locally."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def describe(values):
    x = np.asarray(values, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=0))
    half = float(student_t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    return dict(mean=mean, sd=sd, n=len(x), mean_ci95=[mean - half, mean + half] if half is not None else None)


def load_records(folders, prefix):
    paths = [p for folder in folders for p in sorted((BASE / folder).glob(prefix + '???.json'))]
    rows = [json.loads(p.read_text()) for p in paths]
    for row in rows:
        if not row['complete'] or row['device'] != 'cpu' or row['torch_threads'] != 1 or row['cuda_visibility'] != '':
            raise AssertionError('A completed, single-thread CPU record is required')
        if row['splits']['calibration_used_for_encoder'] or row['splits']['calibration_used_for_head']:
            raise AssertionError('Calibration examples must be excluded from fitting')
        if len({b['minibatch_order_sha256'] for b in row['branches']}) != 1:
            raise AssertionError('Unmatched future minibatches')
        if any(not b.get('matches_primary_only_exactly', True) for b in row['branches']):
            raise AssertionError('Detached control identity failed')
    return rows, paths


def experiment_summary(rows):
    if not rows:
        return None
    if len({r['seed'] for r in rows}) != len(rows):
        raise AssertionError('Duplicate seeds')
    out = dict(n=len(rows), seeds=sorted(r['seed'] for r in rows),
               healthy=describe([r['healthy']['frozen']['acc'] for r in rows]),
               healthy_coverage=describe([r['healthy']['frozen']['coverage'] for r in rows]),
               branches=[])
    for i, template in enumerate(rows[0]['branches']):
        bs = [r['branches'][i] for r in rows]
        if any((b['mode'], b['alpha']) != (template['mode'], template['alpha']) for b in bs):
            raise AssertionError('Branch ordering mismatch')
        end = [b['trajectory'][-1] for b in bs]
        result = dict(mode=template['mode'], alpha=template['alpha'],
                      frozen=describe([z['frozen']['acc'] for z in end]),
                      recalibrated=describe([z['recalibrated']['acc'] for z in end]),
                      retrained=describe([z['retrained']['acc'] for z in end]),
                      own=describe([z['own']['acc'] for z in end]),
                      relative_trace=describe([z['geometry']['relative_trace'] for z in end]),
                      relative_scatter=describe([z['geometry']['relative_scatter'] for z in end]),
                      scalar_residual=describe([z['geometry']['scalar_fit_relative_residual'] for z in end]),
                      frozen_nll=describe([z['frozen']['nll'] for z in end]),
                      frozen_ece=describe([z['frozen']['ece'] for z in end]),
                      frozen_coverage=describe([z['frozen']['coverage'] for z in end]),
                      gate_rate=float(np.mean([b['gate']['gate'] for b in bs])))
        primary = [r['branches'][0]['trajectory'][-1] for r in rows]
        result['paired_frozen_difference_from_primary'] = describe([
            z['frozen']['acc'] - p['frozen']['acc'] for z, p in zip(end, primary)])
        audits = [a for b in bs for a in b['update_audits']]
        result['update_audit'] = dict(n=len(audits),
            flow_step_sign_disagreement=sum(a['gradient_flow_dV'] * a['actual_delta_V'] < 0 for a in audits),
            actual_contracting=sum(a['actual_delta_V'] < 0 for a in audits),
            auxiliary_counterfactual_contracting=sum(a['auxiliary_counterfactual_delta_V'] < 0 for a in audits),
            negative_c_but_auxiliary_counterfactual_contracting=sum(a['c'] < 0 and a['auxiliary_counterfactual_delta_V'] < 0 for a in audits))
        out['branches'].append(result)
    return out


def branch(summary, mode, dose):
    return next(b for b in summary['branches'] if b['mode'] == mode and b['alpha'] == dose)


def main():
    har, hpaths = load_records(['har'], 'h')
    replication, vpaths = load_records(['replication'], 'v')
    real, rpaths = load_records(['real'], 'r')
    projected, ppaths = load_records(['projected'], 'p')
    summary = dict(
        protocol='independent calibration; one cloned healthy model and Adam state per seed',
        confidence_intervals='Descriptive seed-level t intervals conditional on the fixed dataset and recipe; not dataset-population inference.',
        har=experiment_summary(har + replication), initial_har=experiment_summary(har),
        heldout_seed_replication=experiment_summary(replication),
        real=experiment_summary(real), projected=experiment_summary(projected),
        input_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in hpaths + vpaths + rpaths + ppaths})
    (BASE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    macros = {}
    h = summary['har']
    macros.update(CpuHarN=str(h['n']), CpuHarHealthy=f"{h['healthy']['mean']:.3f}",
                  CpuHarPrimary=f"{branch(h, 'primary', 0)['frozen']['mean']:.3f}")
    for dose, suffix in [(.1, 'Low'), (3., 'Mid'), (100., 'High')]:
        b = branch(h, 'joint', dose)
        for key, label in [('frozen', 'Joint'), ('recalibrated', 'Recal'), ('retrained', 'Retrain'), ('own', 'Own')]:
            macros['CpuHar' + label + suffix] = f"{b[key]['mean']:.3f}"
        macros['CpuHarTrace' + suffix] = f"{b['relative_trace']['mean']:.4f}"
    if summary['real']:
        r = summary['real']
        macros.update(CpuRmlN=str(r['n']), CpuRmlHealthy=f"{r['healthy']['mean']:.3f}",
                      CpuRmlPrimary=f"{branch(r,'primary',0)['frozen']['mean']:.3f}")
        for dose, suffix in [(3., 'Three'), (10., 'Ten'), (30., 'Thirty')]:
            b = branch(r, 'joint', dose)
            for key, label in [('frozen', 'Joint'), ('recalibrated', 'Recal'), ('retrained', 'Retrain'), ('own', 'Own'), ('relative_scatter','Scatter')]:
                macros['CpuRml' + label + suffix] = f"{b[key]['mean']:.3f}"
        audits = [branch(r, 'joint', dose)['update_audit'] for dose in [3.,10.,30.]]
        macros['CpuRmlAuditN'] = str(sum(a['n'] for a in audits))
        macros['CpuRmlAuditDisagree'] = str(sum(a['flow_step_sign_disagreement'] for a in audits))
    (ROOT / 'paper/cpu_evidence_numbers.tex').write_text(
        '% Generated locally from retrieved CPU records. Do not edit numbers manually.\n' +
        '\n'.join('\\newcommand{\\' + name + '}{' + value + '}' for name, value in macros.items()) + '\n', encoding='utf-8')

    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Paired checkpoint continuation with held-out calibration. Means $\pm$ population seed standard deviations. Each branch starts from the same healthy model and Adam state; the future minibatch order is shared. Trace is relative to the initial healthy checkpoint. HAR uses 60 healthy plus 30 continuation epochs; RadioML uses 25 plus 15. The real auxiliary head is trained on detached features during warm-up.}',
             r'\label{tab:cpu_paired}', r'\footnotesize',
             r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Case / dose & arm & frozen & input recal. & retrained & own & rel. trace \\', r'\midrule']
    for label, item, doses in [('HAR variance', h, [.1,3.,100.]),
                                ('HAR projected', summary['projected'], [.1,3.,100.]),
                                ('RadioML SNR', summary['real'], [3.,10.,30.])]:
        if item is None:
            continue
        lines.append(r'\multicolumn{7}{l}{\textit{' + label + ', ' + str(item['n']) + r' seeds}} \\')
        def row_text(left, b):
            cells = [('$%.3f\\pm%.3f$' % (b[k]['mean'], b[k]['sd'])) for k in ['frozen','recalibrated','retrained','own']]
            return left + ' & ' + ' & '.join(cells) + (' & $%.4f$ \\\\' % b['relative_trace']['mean'])
        lines.append(row_text(r'$0$ & primary only', branch(item, 'primary', 0)))
        for dose in doses:
            lines.append(row_text('%g & joint' % dose, branch(item, 'joint', dose)))
            lines.append(row_text('%g & STF-cal' % dose, branch(item, 'stfcal', dose)))
        lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines.pop()
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (ROOT / 'paper/cpu_evidence_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    audits = [r'\begin{table}[t]',r'\centering',
              r'\caption{Actual-step audit on a fixed 256-example fitting batch, using the first minibatch of predeclared continuation epochs. Disagree counts opposite signs for the gradient-flow derivative and actual Adam variance change. Aux. contracts compares the joint step with a primary-only Adam step from the same optimizer state. The final column counts negative flow auxiliary flux with a contracting auxiliary counterfactual. Checkpoints within a trajectory are dependent.}',
              r'\label{tab:cpu_adam}',r'\footnotesize',r'\begin{tabular}{lrrrrr}',r'\toprule',
              r'Case & $\alpha$ & updates & disagree & aux. contracts & $c<0$, aux. contracts \\',r'\midrule']
    for label, item, doses in [('HAR variance',h,[.1,3.,100.]),
                               ('HAR projected',summary['projected'],[.1,3.,100.]),
                               ('RadioML SNR',summary['real'],[3.,10.,30.])]:
        if item is None:
            continue
        for dose in doses:
            a=branch(item,'joint',dose)['update_audit']
            audits.append('%s & %g & %d & %d & %d & %d \\\\' % (label,dose,a['n'],a['flow_step_sign_disagreement'],a['auxiliary_counterfactual_contracting'],a['negative_c_but_auxiliary_counterfactual_contracting']))
    audits.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'paper/cpu_adam_table.tex').write_text('\n'.join(audits)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                        'font.size':9.5,'axes.labelsize':10,'axes.titlesize':9.5,
                        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
                        'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(6.9,2.7))
    all_har = har + replication
    def curves(mode, field, dose):
        return np.array([[p[field]['acc'] for p in next(b for b in row['branches'] if b['mode']==mode and b['alpha']==dose)['trajectory']] for row in all_har])
    epochs = [z['epoch'] for z in branch_record(all_har[0], 'joint', 3)['trajectory']]
    for mode, field, dose, color, name in [('joint','frozen',3.,'#C44E52','joint, frozen'),
                                          ('joint','recalibrated',3.,'#55A868','joint, input recal.'),
                                          ('detach','frozen',3.,'#4C72B0','detached, frozen')]:
        values=curves(mode,field,dose)
        axes[0].plot(epochs,values.mean(0),color=color,lw=1.7,label=name)
        axes[0].fill_between(epochs,values.mean(0)-values.std(0),values.mean(0)+values.std(0),color=color,alpha=.12,lw=0)
    axes[0].set(xlabel='epochs after shared checkpoint',ylabel='test accuracy',ylim=(.2,1.0),
                title=r'(a) paired HAR continuation, $\alpha=3$')
    axes[0].legend(loc='lower right',frameon=False)
    doses=[3,10,30,100,300,3000]
    archived=[]
    for dose in doses:
        name='rml_snr_readout_a%d.json'%dose
        archived.append(json.loads((ROOT/'results/READOUT'/name).read_text())['rows'])
    for key,color,name in [('frozen','#C44E52','frozen'),('retrained','#55A868','retrained')]:
        values=np.array([[r['readouts']['joint'][key]['acc'] for r in rows] for rows in archived])
        axes[1].plot(doses,values.mean(1),'-o',color=color,lw=1.5,ms=3,label=name)
        axes[1].fill_between(doses,values.mean(1)-values.std(1),values.mean(1)+values.std(1),color=color,alpha=.12,lw=0)
    detached=np.mean([r['readouts']['stf0']['frozen']['acc'] for r in archived[0]])
    axes[1].axhline(detached,color='#4C72B0',lw=1.5,label='detached, frozen')
    axes[1].axhline(1/11,color='#777777',lw=.8,ls=':',label='chance')
    axes[1].set(xscale='log',xlabel=r'auxiliary dose $\alpha$',ylabel='test accuracy',ylim=(.05,.38),
                title='(b) historical SNR dose scan')
    axes[1].legend(loc='upper right',frameon=False,fontsize=7.5)
    fig.tight_layout(pad=.5,w_pad=1.0)
    output=ROOT/'paper/figures/fig_paired_and_dose'
    fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'),dpi=220,bbox_inches='tight')
    print(json.dumps(dict(har_n=h['n'],real_n=summary['real']['n'] if summary['real'] else 0,
                          projected_n=summary['projected']['n'] if summary['projected'] else 0,
                          har_mid=branch(h,'joint',3),macros=macros),indent=2))


def branch_record(record, mode, dose):
    return next(b for b in record['branches'] if b['mode']==mode and b['alpha']==dose)


if __name__=='__main__':
    main()
