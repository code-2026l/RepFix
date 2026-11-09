"""Render editable layered feature diagrams; shapes encode no measurements."""
from pathlib import Path
import matplotlib.pyplot as plt
from figure_style import WIDTH,apply_style,save
from spatial_diagrams import consumer_planes,nullspace_planes,validate_layout

ROOT=Path(__file__).resolve().parents[2]


def consumer_interface(ax):
    consumer_planes(ax)


def nullspace_control():
    apply_style()
    fig=plt.figure(figsize=(WIDTH,2.5))
    ax=fig.add_axes([.008,.015,.984,.97])
    nullspace_planes(ax)
    checked=validate_layout(fig)
    result=save(fig,ROOT/'paper/figures/fig_nullspace_control')
    result['checked_module_labels']=checked
    plt.close(fig)
    return result


if __name__=='__main__':nullspace_control()
