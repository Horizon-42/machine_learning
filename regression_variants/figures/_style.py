"""Shared plotting style for the Regression Variants tutorial figures.

Import this at the top of every figure script:

    from _style import NAVY, ORANGE, GREEN, RED, PURPLE, GRAY, PALETTE, setup, save

`setup()` configures a clean, consistent matplotlib style so every figure in the
bundle looks like it belongs to the same set.
"""
import matplotlib
matplotlib.use("Agg")            # no display needed; we only save files
import matplotlib.pyplot as plt

# --- colour palette (roughly matching the THWS deck) -----------------------
NAVY   = "#1F4E79"   # primary / squared-error / L2
ORANGE = "#E8873A"   # accent / highlight
GREEN  = "#2E8B57"   # absolute-error / L1
RED    = "#C0392B"   # outlier / warning
PURPLE = "#7E57C2"   # third series
GRAY   = "#8A8A8A"   # secondary lines
PALETTE = [NAVY, ORANGE, GREEN, RED, PURPLE, "#17A2B8", "#B5179E", "#8B5E00"]


def setup():
    plt.rcParams.update({
        "figure.dpi":        130,
        "savefig.dpi":       130,
        "savefig.bbox":      "tight",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.titleweight":  "bold",
        "axes.labelsize":    13,
        "axes.edgecolor":    "#555555",
        "axes.linewidth":    1.1,
        "axes.grid":         True,
        "grid.color":        "#DDDDDD",
        "grid.linewidth":    0.8,
        "legend.fontsize":   11.5,
        "legend.frameon":    True,
        "legend.framealpha": 0.95,
        "lines.linewidth":   2.4,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
