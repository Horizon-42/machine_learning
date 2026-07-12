"""Figure 1 - What each penalty 'charges' a single weight.

Plots the L0, L1 and L2 penalty on ONE weight w_j as a function of its value.
This is the picture behind the slide question "what is the penalty for
w_j = 0.00001?".  It shows *why* L1 and L0 create exact zeros while L2 does not.
"""
import numpy as np
from _style import NAVY, ORANGE, GREEN, setup, save
import matplotlib.pyplot as plt

setup()
lam = 1.0
w = np.linspace(-3, 3, 1000)

L2 = 0.5 * lam * w**2          # (lambda/2) * w^2
L1 = lam * np.abs(w)           # lambda * |w|
# L0 = lambda for any non-zero weight, 0 exactly at 0
L0 = np.where(np.abs(w) < 1e-9, 0.0, lam)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

for ax, y, title, sub, col in [
    (axes[0], L0, r"$L_0$ penalty:  $\lambda\,\mathbb{1}[w_j\neq 0]$",
     "constant charge for being non-zero", NAVY),
    (axes[1], L1, r"$L_1$ penalty:  $\lambda\,|w_j|$",
     "charge stays proportional near 0", GREEN),
    (axes[2], L2, r"$L_2$ penalty:  $\frac{\lambda}{2}\,w_j^2$",
     "charge vanishes near 0", ORANGE),
]:
    if title.startswith(r"$L_0$"):
        # draw the step with an open/closed marker so the jump is clear
        ax.plot(w[w < 0], y[w < 0], color=col)
        ax.plot(w[w > 0], y[w > 0], color=col)
        ax.plot(0, 0, "o", color=col, ms=8)                 # value AT zero
        ax.plot(0, lam, "o", mfc="white", mec=col, ms=8)    # limit approaching 0
    else:
        ax.plot(w, y, color=col)
    ax.axvline(0, color="#BBBBBB", lw=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(r"$w_j$")
    ax.text(0.5, 0.92, sub, transform=ax.transAxes, ha="center",
            va="top", fontsize=11, style="italic", color="#444444")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.15, 2.6)

axes[0].set_ylabel("penalty")
fig.suptitle("How much does each regularizer charge one weight?",
             fontsize=15, fontweight="bold", y=1.02)
save(fig, "fig1_norm_penalties.png")
