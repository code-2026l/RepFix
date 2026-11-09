"""Confirmation wrapper: the development driver is kept byte-identical."""
import argparse
import hashlib
import json
from pathlib import Path

import posthoc_chart_cpu as calculation
import paired_readout_continuation as paired


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['radioml'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seed not in range(700, 710):
        parser.error('Only the ten predeclared confirmation seeds are allowed')
    args.phase = 'confirmation'
    args.warm_epochs = args.cont_epochs = args.limit_fit = args.limit_eval = 0
    calculation.run(args)
    record = json.loads(args.out.read_text())
    record.update(selection_sha256=args.selection_sha256,
                  confirmation_wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    paired.save_json(args.out, record)
