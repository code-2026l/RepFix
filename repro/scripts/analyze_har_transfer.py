"""Replay the fixed HAR screen, fit the forecast, and score confirmation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_early_update import digest, interval
from paired_readout_continuation import state_hash

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'docs/plans/2026-11-11-har-utility-and-risk-transfer.md'
ADDENDUM = ROOT / 'docs/plans/2026-11-11-har-forecast-reliability.md'
RESULTS = ROOT / 'repro/results/CPUCONT'
CHOICE = RESULTS / 'har_transfer_choice_20261111.json'
FORECASTER = RESULTS / 'har_forecaster_20261111.json'
CONFIRMATION = RESULTS / 'har_transfer_confirmation_20261111.json'
STAGES = dict(screen=('k', 8100, 8), calibration=('d', 8120, 12),
              confirmation=('c', 8200, 16))


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def load(stage, records, arrays):
    prefix, start, count = STAGES[stage]
    for offset in range(count):
        key = f'{prefix}{offset:03d}'
        path = records / f'{key}.json'
        row = json.loads(path.read_text())
        assert row['complete'] and row['device'] == 'cpu' and row['stage'] == stage
        assert row['seed'] == start + offset and row['plan_sha256'] == digest(PLAN)
        assert row['test_loaded'] == (stage == 'confirmation')
        subjects = [set(row[key]) for key in ('fit_subjects', 'monitor_subjects', 'evaluation_subjects')]
        assert not any(a & b for i, a in enumerate(subjects) for b in subjects[i+1:])
        array = arrays / f'{key}_states/replay.pt'
        assert digest(array) == row['replay_sha256']
        replay = torch.load(array, map_location='cpu', weights_only=True)
        outputs, labels = replay['outputs'], replay['labels']
        primary = outputs if stage == 'screen' else outputs['primary']
        assert len(primary) == len(row['outcomes']) and len(labels) == row['evaluation_examples']
        for arm in row['outcomes']:
            name = f"{arm['objective']}/{arm['dose']}" if stage == 'screen' else arm['key']
            accuracy = (primary[name].argmax(1) == labels).double().mean().item()
            assert accuracy == arm['accuracy']
        receipt = None
        if stage != 'screen':
            receipt_path = path.with_suffix('.forecast.json')
            assert digest(receipt_path) == row['forecast_sha256']
            receipt = json.loads(receipt_path.read_text())
            assert receipt['seed'] == row['seed'] and receipt['plan_sha256'] == row['plan_sha256']
            assert receipt['early_epoch'] == 3 and not receipt['test_loaded']
            assert receipt['choice_sha256'] == digest(CHOICE)
            assert receipt['forecaster_sha256'] == (digest(FORECASTER) if stage == 'confirmation' else None)
            assert receipt['early_sha256'] == state_hash(tuple(outputs[k] for k in ('old_early', 'early', 'thresholds')))
            assert all(torch.equal(outputs['old'][head], outputs['retained'][head]) for head in outputs['old'])
        yield row, outputs, receipt


def screen(rows):
    values = np.array([
        [[100 * (a['accuracy'] - r['outcomes'][0]['accuracy']),
          a['auxiliary'][a['objective']] / r['outcomes'][0]['auxiliary'][a['objective']]]
         for a in r['outcomes'][1:]] for r, _, _ in rows])
    candidates = []
    for arm, measurements in zip(rows[0][0]['outcomes'][1:], values.transpose(1, 0, 2)):
        gain, ratio = measurements.mean(0).tolist()
        candidates.append(dict(objective=arm['objective'], dose=arm['dose'], gain_pp=gain,
                               auxiliary_ratio=ratio, qualifies=gain >= .5 and ratio <= .98))
    eligible = [row for row in candidates if row['qualifies']]
    selected = max(eligible, key=lambda row: row['gain_pp']) if eligible else candidates[1]
    return dict(passed=bool(eligible), fallback=not eligible, selected=selected,
                plan_sha256=digest(PLAN), addendum_sha256=digest(ADDENDUM), candidates=candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--arrays', type=Path)
    args = parser.parse_args()
    arrays = args.arrays or ROOT / f'tmp/conditional_contrastivehar{args.stage}'
    records = args.records or arrays
    torch.set_num_threads(4)
    rows = list(load(args.stage, records, arrays))
    if args.stage == 'screen':
        result, output = screen(rows), CHOICE
    else:
        from har_early_forecast import calibrate, confirm
        if args.stage == 'calibration':
            result, output = calibrate(rows), FORECASTER
        else:
            result, output = confirm(rows), CONFIRMATION
    save(output, result)
    print(json.dumps({key: val for key, val in result.items() if key not in ('model', 'references')},
                     indent=2))


if __name__ == '__main__':
    main()
