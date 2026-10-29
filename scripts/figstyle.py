"""Shared matplotlib style for all paper figures.

Palette: validated categorical set (fixed slot order, never cycled).
Class / series color assignment is FIXED across every figure in the paper.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# validated categorical palette (light surface)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#4a3aa7", "#e34948"]

# fixed semantic assignment (entity -> color, used everywhere)
C_NORMAL = PAL[0]      # blue
C_NATURAL = PAL[2]     # aqua
C_FDIA = PAL[7]        # red
C_REPLAY = PAL[6]      # violet
C_RFDI = PAL[1]        # orange
CLASS_COLORS = [C_NORMAL, C_NATURAL, C_FDIA, C_REPLAY, C_RFDI]
CLASS_NAMES = ["Normal", "Natural", "FDIA", "Replay", "RFDI"]

SEQ = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]   # sequential blue ramp
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def setup():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": INK2, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax
