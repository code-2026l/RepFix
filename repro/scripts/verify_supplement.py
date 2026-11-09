"""Verify a pristine supplementary release; run optional checks in a temporary copy.

The default path uses only Python's standard library. It checks files and recorded
provenance, not predictions from tensor checkpoints excluded from this archive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    'test_consumer_risk.py', 'test_readout_alignment.py',
    'test_scale_boundaries.py', 'test_readout_nullspace.py',
    'test_compiled_readout.py', 'test_nonlinear_interface.py',
    'test_nonlinear_selection.py',
]
PLANS = {
    'benefit': '2026-11-09-prospective-benefit-compatibility.md',
    'compiled': '2026-11-09-compiled-interface.md',
    'nonlinear': '2026-11-09-nonlinear-interface.md',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
        raise ValueError('Unsafe manifest path: ' + name)
    target = root.joinpath(*path.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Payload path leaves the artifact: ' + name)
    return target


def verify_files(root):
    manifest = json.loads((root / 'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
    files = manifest['sha256']
    if not files:
        raise ValueError('Empty artifact manifest')
    for name, expected in files.items():
        path = payload_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed payload: ' + name)
    tex = (root / 'paper/RepFix_ICLR2027.tex').read_text(encoding='utf-8')
    inputs = re.findall(r'\\input\{([^}]+)\}', tex)
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
    for name in inputs + figures:
        if 'paper/' + name not in files:
            raise ValueError('Unmanifested manuscript input: ' + name)
    if len(figures) != 17:
        raise ValueError('Expected the 17 included manuscript figures')
    if manifest['pdf_sha256'] != digest(root / 'paper/RepFix_ICLR2027.pdf'):
        raise ValueError('The manuscript PDF does not match the manifest')

    sources = plans = records = 0
    python_hashes = {value for name, value in files.items() if name.endswith('.py')}
    base = 'repro/results/CPUCONT/'
    for name in sorted(files):
        if not name.startswith(base) or not name.endswith('.json'):
            continue
        row = json.loads((root / name).read_text(encoding='utf-8'))
        if not isinstance(row, dict):
            continue
        if row.get('complete'):
            records += 1
        source_map = row.get('source_sha256', {})
        if isinstance(source_map, str):
            if source_map not in python_hashes:
                raise ValueError('Recorded driver digest is unavailable: ' + name)
            sources += 1
            source_map = {}
        for field, expected in row.items():
            if field.endswith('_source_sha256') and isinstance(expected, str):
                if expected not in python_hashes:
                    raise ValueError('Recorded source digest is unavailable: ' + name + '/' + field)
                sources += 1
        for source, expected in source_map.items():
            if not isinstance(expected, str) or len(expected) != 64:
                continue
            path = payload_path(root, source)
            if not path.is_file():
                path = payload_path(root, 'repro/scripts/' + source)
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Recorded source mismatch in ' + name + ': ' + source)
            sources += 1
        if 'plan_sha256' in row:
            families = [family for family in PLANS if family in name]
            if len(families) == 1:
                path = root / 'docs/plans' / PLANS[families[0]]
                if digest(path) != row['plan_sha256']:
                    raise ValueError('Recorded plan mismatch: ' + name)
                plans += 1

    choice = json.loads((root / base / 'nonlinear_development_choice.json').read_text(encoding='utf-8'))
    if any(v['confirm'] for v in choice['selected'].values()):
        raise ValueError('The nonlinear decision differs from the released failed gate')
    development = [name for name in files if re.fullmatch(base + r'nonlinear_development/b\d{3}\.json', name)]
    if len(development) != 6 or any(name.startswith(base + 'nonlinear_confirmation/') for name in files):
        raise ValueError('Unexpected nonlinear development/confirmation family')
    for name in development:
        row = json.loads((root / name).read_text(encoding='utf-8'))
        if row['splits']['test_metrics_computed'] or any('test' in b for b in row['branches']):
            raise ValueError('Development record contains test outcomes: ' + name)

    return manifest, dict(payload_files=len(files), manuscript_figures=len(figures),
        numerical_inputs=len(inputs), completed_cpu_records=records,
        recorded_source_hashes_checked=sources, recorded_plan_hashes_checked=plans,
        nonlinear_development_records=len(development), nonlinear_confirmation_tasks=0,
        manuscript_pdf_sha256=manifest['pdf_sha256'])


def execute(command, root):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUTF8='1')
    start = time.perf_counter()
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=300)
    if result.returncode:
        raise RuntimeError('Check failed: ' + ' '.join(command[1:]) + '\n' +
                           result.stdout[-4000:] + '\n' + result.stderr[-2000:])
    return result.stdout, round(time.perf_counter() - start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', action='store_true', help='Run 47 structural checks on CPU')
    parser.add_argument('--figures', action='store_true', help='Rebuild all included figures and compare numerical artists')
    parser.add_argument('--paper', action='store_true', help='Rebuild the manuscript and compare page text and pixels')
    args = parser.parse_args()
    manifest, report = verify_files(ROOT)
    report['checks'] = {}
    report['trained_tensor_states_replayed'] = False
    if not (args.tests or args.figures or args.paper):
        print(json.dumps(report, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix='repfix-review-') as temporary:
        copy = Path(temporary)
        for name in manifest['sha256']:
            target = payload_path(copy, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copyfile(ROOT / 'ARTIFACT_MANIFEST.json', copy / 'ARTIFACT_MANIFEST.json')
        if args.tests:
            print('Running structural checks on CPU...', flush=True)
            stdout, elapsed = execute([sys.executable, '-m', 'pytest'] +
                ['repro/scripts/' + name for name in TESTS] + ['-q'], copy)
            passed = re.search(r'(\d+) passed', stdout)
            if not passed or int(passed.group(1)) != 47:
                raise ValueError('Unexpected focused-test result: ' + stdout[-1000:])
            report['checks']['structural_tests'] = dict(passed=47, elapsed_seconds=elapsed)
        if args.figures:
            print('Rebuilding all 17 figures from recorded results...', flush=True)
            reference = json.loads((ROOT / 'paper/figures/publication_figure_manifest.json').read_text())
            _, elapsed = execute([sys.executable, 'repro/scripts/render_paper_figures.py'], copy)
            rebuilt = json.loads((copy / 'paper/figures/publication_figure_manifest.json').read_text())
            if set(reference['figures']) != set(rebuilt['figures']):
                raise ValueError('The figure set changed')
            for name, original in reference['figures'].items():
                for field in ['width_inches', 'height_inches', 'numerical_artists_sha256']:
                    if original[field] != rebuilt['figures'][name][field]:
                        raise ValueError('Figure evidence differs: ' + name + '/' + field)
            report['checks']['figures'] = dict(rebuilt=17, matching_numerical_artists=17,
                elapsed_seconds=elapsed)
        if args.paper:
            print('Building and comparing the 62-page manuscript...', flush=True)
            _, elapsed = execute([sys.executable, 'repro/scripts/build_cpu_paper.py'], copy)
            build = json.loads((copy / 'tmp/cpu_build/build_report.json').read_text())
            if build['warnings'] or build['main_text_pages'] != 9 or build['total_pages'] != 62:
                raise ValueError('Unexpected manuscript build: ' + json.dumps(build))
            import fitz
            with fitz.open(ROOT / 'paper/RepFix_ICLR2027.pdf') as old, \
                    fitz.open(copy / 'paper/RepFix_ICLR2027.pdf') as new:
                if len(old) != len(new) or len(new) != 62:
                    raise ValueError('Manuscript page count changed')
                text_matches = pixel_matches = 0
                for a, b in zip(old, new):
                    text_matches += a.get_text() == b.get_text()
                    pixel_matches += a.get_pixmap(alpha=False).samples == b.get_pixmap(alpha=False).samples
            if text_matches != 62 or pixel_matches != 62:
                raise ValueError('Rendered manuscript differs: text %s/62, pixels %s/62. '
                    'Check the LaTeX and figure font environment.' % (text_matches, pixel_matches))
            report['checks']['paper'] = dict(main_text_pages=9,total_pages=62,warnings=[],
                matching_page_text=62,matching_page_pixels_at_72dpi=62,elapsed_seconds=elapsed)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
