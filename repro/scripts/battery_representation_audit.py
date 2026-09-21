"""Run the existing battery recipe with explicit across-example diagnostics.

The base driver and its legacy scatter remain unchanged. Extra diagnostics are
computed after it restores the selected checkpoint, inside an RNG fork. This
does not repair the base driver's checkpoint-selection protocol.
"""
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
import battery_bmtl_v3 as base


@torch.no_grad()
def representation_stats(model, loader, device):
    n = 0
    total = outer = None
    preds = []
    for xb, *_ in loader:
        z = model.embed(xb.to(device, non_blocking=True))
        zd = z.double()
        if total is None:
            total = zd.sum(0)
            outer = zd.T @ zd
        else:
            total += zd.sum(0)
            outer += zd.T @ zd
        n += len(z)
        preds.append(torch.stack((model.head_soh(z).flatten(),
                                  model.head_rul(z).flatten()), dim=1).cpu().double())
    cov = (outer - torch.outer(total, total) / n) / max(n - 1, 1)
    eig = torch.linalg.eigvalsh((cov + cov.T)/2).clamp_min(0)
    trace = eig.sum()
    probs = eig / trace.clamp_min(1e-30)
    pred = torch.cat(preds)
    return dict(n=n, between_sample_trace=float(trace),
                mean_feature_variance=float(trace / len(eig)),
                participation_rank=float(trace.square()/eig.square().sum().clamp_min(1e-30)),
                entropy_rank=float((-(probs * probs.clamp_min(1e-30).log()).sum()).exp()),
                covariance_eigenvalues=eig.cpu().tolist(),
                prediction_variance_normalized_units=pred.var(dim=0).tolist())


def main():
    original_run = base.run
    diagnostics = []

    def audited_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        model, loader, device = args[0], args[2], args[5]
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            row = representation_stats(model, loader, device)
        row.update(call_index=len(diagnostics), single=kwargs.get('single', False),
                   stf_mode=kwargs.get('stf_mode', 'off'), stf=bool(args[7]),
                   legacy_coordinate_variance=result['scatter'])
        diagnostics.append(row)
        result['representation_diagnostics'] = row
        print('[representation audit]', {k:v for k,v in row.items() if k != 'covariance_eigenvalues'}, flush=True)
        return result

    base.run = audited_run
    base.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    d = json.loads(out.read_text())
    d['representation_audit'] = dict(checkpoint='base driver selected checkpoint',
        warning='legacy driver selects best checkpoint using its evaluation split',
        calls=diagnostics)
    out.write_text(json.dumps(d, indent=2))


if __name__ == '__main__':
    main()
