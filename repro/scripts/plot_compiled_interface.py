"""Local vector diagram of the constrained-last-affine deployment contract."""
from pathlib import Path

import matplotlib.pyplot as plt

from figure_style import WIDTH, apply_style, save
from spatial_diagrams import compiled_planes, validate_layout

ROOT = Path(__file__).resolve().parents[2]


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH, 2.65))
    ax = fig.add_axes([.008, .015, .984, .97])
    compiled_planes(ax)
    validate_layout(fig)
    result = save(fig, ROOT / 'paper/figures/fig_compiled_control')
    plt.close(fig)
    return result


if __name__ == '__main__':
    main()
