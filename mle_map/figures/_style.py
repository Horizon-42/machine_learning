"""Shared plotting style for the MLE / MAP tutorial figures.

Import at the top of every figure script:
    from _style import NAVY, ORANGE, GREEN, RED, PURPLE, GRAY, PALETTE, setup, save
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY   = "#1F4E79"
ORANGE = "#E8873A"
GREEN  = "#2E8B57"
RED    = "#C0392B"
PURPLE = "#7E57C2"
GRAY   = "#8A8A8A"
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
