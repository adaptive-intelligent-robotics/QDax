import math
import os
import tempfile

import matplotlib.pyplot as plt


def get_fig_size(fig_width_cm, fig_height_cm=None):
    """Convert dimensions in centimeters to inches.
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_height_cm:
        golden_ratio = (1 + math.sqrt(5)) / 2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return list(map(lambda x: x / 2.54, size_cm))


"""
The following functions can be used by scripts to get the sizes of
the various elements of the figures.
"""


def label_size():
    """Size of axis labels"""
    return 10


def font_size():
    """Size of all texts shown in plots"""
    return 10


def ticks_size():
    """Size of axes' ticks"""
    return 8


def axis_lw():
    """Line width of the axes"""
    return 0.6


def plot_lw():
    """Line width of the plotted curves"""
    return 1.5


def figure_setup():
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {
        "text.usetex": False,
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        "figure.dpi": 200,
        "font.size": font_size(),
        "font.sans-serif": ["Helvetica"],
        "font.monospace": [],
        "axes.labelsize": label_size(),
        "axes.titlesize": font_size(),
        "axes.linewidth": axis_lw(),
        # 'text.fontsize': font_size(),
        "legend.fontsize": font_size(),
        "xtick.labelsize": ticks_size(),
        "ytick.labelsize": ticks_size(),
        "font.family": "sans-serif",
        "font.serif": [],
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,
        "grid.linestyle": "--",
        "svg.fonttype": "none",
    }

    # sns.set(font="Palatino")

    plt.rcParams.update(params)


def set_font_size(new_font_size, tick_size=None):
    font_size_params = {
        "font.size": new_font_size,
        "axes.titlesize": new_font_size,
        "legend.fontsize": new_font_size,
        "axes.labelsize": new_font_size,
    }
    if tick_size:
        font_size_params.update(
            {
                "xtick.labelsize": tick_size,
                "ytick.labelsize": tick_size,
            }
        )
    plt.rcParams.update(font_size_params)


def archive_ticks_params(ax):
    ax.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        width=1,
        length=1,
    )


def save_fig(fig, file_name, fmt=None, dpi=300, tight=True, transparent=None):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it."""

    if not fmt:
        fmt = file_name.strip().split(".")[-1]

    if fmt not in ["eps", "png", "pdf", "svg"]:
        raise ValueError("unsupported format: %s" % (fmt,))

    extension = ".%s" % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches="tight", transparent=transparent)
    else:
        fig.savefig(tmp_name, dpi=dpi, transparent=transparent)

    print(tmp_name)
    # trim it
