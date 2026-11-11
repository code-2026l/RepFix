"""Publication-width scientific figure style; local rendering only."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import json
import re

WIDTH = 5.5
EXPORTS = {}
FONT = 'Arial'
COLORS = dict(current='#416B8D', frozen='#BB4B64', repair='#14858A',
              reference='#89949E', purple='#8974A2', amber='#B98338',
              ink='#263746', grid='#E7EBEF')


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif':[FONT,'Helvetica','DejaVu Sans'],
        'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.labelsize': 8.1, 'axes.titlesize': 8.1,
        'xtick.labelsize': 7.7, 'ytick.labelsize': 7.7, 'legend.fontsize': 7.5,
        'axes.linewidth': .55, 'lines.linewidth': 1.35, 'lines.markersize': 3.5,
        'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['ink'],
        'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
        'axes.edgecolor': '#A6AFB7', 'axes.axisbelow': True,
        'grid.color': COLORS['grid'], 'grid.linewidth': .45,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    })


def panel(ax, letter, heading):
    ax.set_title(f'$\\mathbf{{{letter}}}$  {heading}',loc='left',pad=8,
                 fontfamily=FONT,fontweight='normal')


def numerical_artists(fig):
    """Hash plotted coordinates, uncertainty paths, areas and heat-map values."""
    payload = []
    for ax in fig.axes:
        data = dict(lines=[(np.asarray(l.get_xdata()).tolist(), np.asarray(l.get_ydata()).tolist())
                           for l in ax.lines], collections=[], images=[], rectangles=[])
        for c in ax.collections:
            data['collections'].append(dict(offsets=np.asarray(c.get_offsets()).tolist(),
                array=np.asarray(c.get_array()).tolist() if c.get_array() is not None else None,
                sizes=np.asarray(c.get_sizes()).tolist() if hasattr(c, 'get_sizes') else None,
                paths=[p.vertices.tolist() for p in c.get_paths()]))
        for p in ax.patches:
            if isinstance(p, matplotlib.patches.Rectangle):
                data['rectangles'].append([p.get_x(), p.get_y(), p.get_width(), p.get_height()])
        data['images'] = [np.asarray(i.get_array()).tolist() for i in ax.images]
        payload.append(data)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()


def polish(fig):
    """Unify legacy text at final width without modifying scientific artists."""
    before = numerical_artists(fig)
    for t in fig.findobj(Text):
        t.set_fontfamily(FONT)
        t.set_fontsize(max(7.1, t.get_fontsize()))
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.title.set_fontsize(8.1)
        ax._left_title.set_fontsize(8.1)
        ax.tick_params(labelsize=7.2)
        ax.spines[['top', 'right']].set_visible(False)
    assert before == numerical_artists(fig), 'A visual change modified plotted evidence'
    return before


def save(fig, output):
    """Export fixed-size vectors and preview, rejecting clipped visible labels."""
    output = Path(output)
    # Legacy generators can set their own font after apply_style(). Keep the
    # complete export family coherent without moving measurement artists.
    before = numerical_artists(fig)
    for label in fig.findobj(Text):
        label.set_fontfamily(FONT)
        match=re.match(r'^\(([a-z])\)\s*(.*)',label.get_text(),flags=re.S)
        if match and label.get_fontsize() >= 8:
            label.set_text(r'$\mathbf{'+match[1]+'}$  '+match[2])
            label.set_fontweight('normal')
    for ax in fig.axes:
        for title in [ax.title,ax._left_title,ax._right_title]:
            title.set_fontweight('normal')
            match=re.match(r'^([a-z])\s{2,}(.*)',title.get_text(),flags=re.S)
            if match:title.set_text(r'$\mathbf{'+match[1]+'}$  '+match[2])
    assert before == numerical_artists(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sizes = []
    excluded = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not ax.axison or not axis.get_visible():
                excluded.update(id(t) for t in axis.findobj(Text))
        for tick in [*ax.xaxis.get_major_ticks(), *ax.xaxis.get_minor_ticks()]:
            if not min(ax.get_xlim()) <= tick.get_loc() <= max(ax.get_xlim()):
                excluded.update([id(tick.label1), id(tick.label2)])
        for tick in [*ax.yaxis.get_major_ticks(), *ax.yaxis.get_minor_ticks()]:
            if not min(ax.get_ylim()) <= tick.get_loc() <= max(ax.get_ylim()):
                excluded.update([id(tick.label1), id(tick.label2)])
    for t in fig.findobj(Text):
        if id(t) in excluded or not t.get_visible() or not t.get_text().strip():
            continue
        # Matplotlib may retain tick Text objects beyond an explicit axis limit.
        axis = t.axes
        if axis is not None and t in axis.get_xticklabels() + axis.get_yticklabels():
            pos = t.get_position()
            if t in axis.get_xticklabels() and not min(axis.get_xlim()) <= pos[0] <= max(axis.get_xlim()):
                continue
            if t in axis.get_yticklabels() and not min(axis.get_ylim()) <= pos[1] <= max(axis.get_ylim()):
                continue
        bb = t.get_window_extent(renderer)
        if bb.width and bb.height:
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
                raise AssertionError((output.name, t.get_text(), tuple(bb.bounds)))
            sizes.append(t.get_fontsize())
    hashes = {}
    for ext in ('pdf', 'png', 'svg'):
        p = output.with_suffix('.' + ext)
        fig.savefig(p, dpi=320)
        if ext == 'svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines()) + '\n',
                         encoding='utf-8', newline='\n')
        hashes[ext] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert b'/Subtype /Type3' not in output.with_suffix('.pdf').read_bytes()
    result = dict(width_inches=float(fig.get_figwidth()), height_inches=float(fig.get_figheight()),
                  smallest_visible_font_pt=float(min(sizes)), output_sha256=hashes,
                  numerical_artists_sha256=numerical_artists(fig))
    EXPORTS[output.name] = result
    return result
