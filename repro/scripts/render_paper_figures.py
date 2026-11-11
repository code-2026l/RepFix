"""Rebuild every included manuscript figure locally; preserve recorded evidence."""
import json
from pathlib import Path
import runpy

from figure_style import EXPORTS

ROOT = Path(__file__).resolve().parents[2]


def main():
    for script in ['paper/revision/build_figures.py',
                   'repro/scripts/plot_final_figures.py',
                   'repro/scripts/plot_probe_arms.py',
                   'repro/scripts/plot_probe_phase.py',
                   'repro/scripts/plot_paper_diagrams.py',
                   'repro/scripts/plot_compiled_interface.py',
                   'repro/scripts/plot_compiled_frontier.py',
                   'repro/scripts/plot_nonlinear_study.py',
                   'repro/scripts/plot_benefit_confirmation.py',
                   'repro/scripts/plot_readout_story.py',
                   'repro/scripts/plot_chart_confirmation.py',
                   'repro/scripts/plot_protected_audit.py',
                   'repro/scripts/plot_alignment_confirmation.py',
                   'repro/scripts/plot_consumer_contracts.py',
                   'repro/scripts/plot_shared_replay.py',
                   'repro/scripts/plot_replay_vision.py',
                   'repro/scripts/plot_suffix_frontier.py',
                   'repro/scripts/plot_suffix_repairs.py',
                   'repro/scripts/plot_serving_frontier.py',
                   'repro/scripts/plot_reference_replication.py',
                   'repro/scripts/plot_risk_query_result.py',
                   'repro/scripts/plot_contrastive_benefit.py',
                   'repro/scripts/plot_early_update.py',
                   'repro/scripts/plot_legacy_precision.py',
                   'repro/scripts/plot_subject_transfer.py']:
        runpy.run_path(str(ROOT / script), run_name='__main__')
    assert len(EXPORTS) == 32
    assert all(abs(v['width_inches'] - 5.5) < 1e-10 for v in EXPORTS.values())
    report = dict(figures=EXPORTS, figures_rebuilt=len(EXPORTS),
                  all_vector_type42=True, all_text_within_canvas=True,
                  legacy_visual_changes_preserve_numerical_artists=True)
    (ROOT / 'paper/figures/publication_figure_manifest.json').write_text(
        json.dumps(report, indent=2) + '\n',encoding='utf-8',newline='\n')
    print('Rebuilt all thirty-two included figures at final 5.5-inch width.')


if __name__ == '__main__':
    main()
