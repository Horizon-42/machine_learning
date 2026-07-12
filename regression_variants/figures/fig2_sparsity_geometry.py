"""Figure 2 - Why L1 produces exact zeros (the diamond-vs-circle picture).

Elliptical contours = the squared training error, centred on the unconstrained
least-squares solution.  The constraint region is a diamond (L1) or a disk (L2).
The regularized solution is the point INSIDE the region with the smallest error;
equivalently, where the growing error contour first touches the region.

The markers are the *actual* constrained minimizers (found numerically), not
hand-placed: the diamond touches at a CORNER (one weight exactly 0 -> sparsity),
the disk touches on a smooth edge (both weights non-zero).
"""
import numpy as np
from scipy.optimize import minimize
from _style import NAVY, ORANGE, GREEN, RED, setup, save
import matplotlib.pyplot as plt

setup()

beta_hat = np.array([2.6, 0.7])                 # least-squares optimum
A = np.array([[1.0, 0.35], [0.35, 1.0]])        # error-bowl shape

def rss(w):
    d = np.asarray(w) - beta_hat
    return d @ A @ d

def rss_grid(W1, W2):
    d1, d2 = W1 - beta_hat[0], W2 - beta_hat[1]
    return A[0, 0]*d1*d1 + 2*A[0, 1]*d1*d2 + A[1, 1]*d2*d2

g = np.linspace(-1.2, 3.6, 400)
W1, W2 = np.meshgrid(g, g)
Z = rss_grid(W1, W2)
levels = np.linspace(0.3, 14, 9)

# ---- actual constrained minimizers -----------------------------------------
t = 1.5
sol1 = minimize(rss, [0.5, 0.5], method="SLSQP",
                constraints={"type": "ineq",
                             "fun": lambda w: t - (abs(w[0]) + abs(w[1]))}).x
r = 1.5
sol2 = minimize(rss, [0.5, 0.5], method="SLSQP",
                constraints={"type": "ineq",
                             "fun": lambda w: r**2 - (w[0]**2 + w[1]**2)}).x

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

# ---- L1 : diamond -----------------------------------------------------------
ax = axes[0]
ax.contour(W1, W2, Z, levels=levels, colors=NAVY, linewidths=1.0, alpha=0.55)
dia = np.array([[t, 0], [0, t], [-t, 0], [0, -t], [t, 0]])
ax.plot(dia[:, 0], dia[:, 1], color=GREEN, lw=2.6)
ax.fill(dia[:, 0], dia[:, 1], color=GREEN, alpha=0.10)
ax.plot(*beta_hat, "o", color="#888", ms=7)
ax.annotate("least-squares\noptimum", beta_hat, textcoords="offset points",
            xytext=(8, 6), fontsize=10, color="#555")
ax.plot(*sol1, "*", color=RED, ms=22, zorder=5)
ax.annotate(r"$L_1$ solution" + "\n" + r"($w_2 = 0$: sparse!)", sol1,
            textcoords="offset points", xytext=(8, 12), fontsize=11, color=RED)
ax.set_title(r"$L_1$  (LASSO):  diamond has corners")

# ---- L2 : disk --------------------------------------------------------------
ax = axes[1]
ax.contour(W1, W2, Z, levels=levels, colors=NAVY, linewidths=1.0, alpha=0.55)
th = np.linspace(0, 2*np.pi, 300)
ax.plot(r*np.cos(th), r*np.sin(th), color=ORANGE, lw=2.6)
ax.fill(r*np.cos(th), r*np.sin(th), color=ORANGE, alpha=0.10)
ax.plot(*beta_hat, "o", color="#888", ms=7)
ax.annotate("least-squares\noptimum", beta_hat, textcoords="offset points",
            xytext=(8, 6), fontsize=10, color="#555")
ax.plot(*sol2, "*", color=RED, ms=22, zorder=5)
ax.annotate(r"$L_2$ solution" + "\n" + r"(both weights $\neq 0$)", sol2,
            textcoords="offset points", xytext=(10, -24), fontsize=11, color=RED)
ax.set_title(r"$L_2$  (ridge):  disk is smooth")

for ax in axes:
    ax.axhline(0, color="#BBB", lw=1)
    ax.axvline(0, color="#BBB", lw=1)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_xlim(-1.2, 3.6)
    ax.set_ylim(-1.2, 3.6)
    ax.set_aspect("equal")

fig.suptitle("Same error bowl, different constraint shape -> different solution",
             fontsize=15, fontweight="bold", y=1.00)
print("L1 solution:", np.round(sol1, 3), "   L2 solution:", np.round(sol2, 3))
save(fig, "fig2_sparsity_geometry.png")
