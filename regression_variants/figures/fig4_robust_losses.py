"""Figure 4 - Robust regression losses and their 'influence'.

Left  : the loss as a function of a single residual r = w^T x - y.
Right : the derivative of the loss w.r.t. r  (= how hard one point pulls on w).

Squared error's influence grows without bound -> one outlier dominates.
Absolute error has constant influence (+/-1).  Huber is smooth at 0 like the
squared error but has bounded influence like the absolute error -> best of both.
"""
import numpy as np
from _style import NAVY, GREEN, PURPLE, setup, save
import matplotlib.pyplot as plt

setup()
r = np.linspace(-3, 3, 1000)
eps = 1.0

squared = 0.5 * r**2
absolute = np.abs(r)
huber = np.where(np.abs(r) <= eps, 0.5 * r**2, eps * (np.abs(r) - 0.5 * eps))

# derivatives (influence functions)
d_squared = r
d_absolute = np.sign(r)
d_huber = np.where(np.abs(r) <= eps, r, eps * np.sign(r))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

ax = axes[0]
ax.plot(r, squared,  color=NAVY,   label=r"squared  $\frac{1}{2}r^2$")
ax.plot(r, absolute, color=GREEN,  label=r"absolute  $|r|$")
ax.plot(r, huber,    color=PURPLE, label=r"Huber  $h_\epsilon(r),\ \epsilon=1$")
for x in (-eps, eps):
    ax.axvline(x, color="#CCC", lw=1, ls=":")
ax.set_title("Loss vs. residual")
ax.set_xlabel(r"residual  $r_i = w^\top x_i - y_i$")
ax.set_ylabel("loss")
ax.set_ylim(-0.2, 4.2)
ax.legend()

ax = axes[1]
ax.plot(r, d_squared,  color=NAVY,   label=r"squared: $r$ (unbounded)")
ax.plot(r, d_absolute, color=GREEN,  label=r"absolute: $\mathrm{sign}(r)$")
ax.plot(r, d_huber,    color=PURPLE, label=r"Huber: clipped at $\pm\epsilon$")
ax.axhline(0, color="#BBB", lw=1)
for x in (-eps, eps):
    ax.axvline(x, color="#CCC", lw=1, ls=":")
ax.set_title("Influence = how hard one point pulls on the fit")
ax.set_xlabel(r"residual  $r_i$")
ax.set_ylabel(r"$d(\text{loss})/dr$")
ax.set_ylim(-3.2, 3.2)
ax.legend()

fig.suptitle("Robustness comes from bounding a point's influence",
             fontsize=15, fontweight="bold", y=1.00)
save(fig, "fig4_robust_losses.png")
