"""Figure 3 - The Gaussian-noise picture behind the squared loss.

Left : a linear model y = w^T x + noise.  The vertical sticks are the residuals
       (errors) between each point and the line.
Right: collect all those residuals and histogram them -> they look Gaussian.
Assuming the errors are i.i.d. Normal is exactly the assumption that makes
Maximum Likelihood equal to minimizing the squared error.
"""
import numpy as np
from _style import NAVY, ORANGE, GRAY, setup, save
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
n = 160
x = np.linspace(0.5, 9.5, n)
w0, w1 = 1.0, 0.9
noise = rng.randn(n) * 1.1
y = w0 + w1 * x + noise
yline = w0 + w1 * x

setup()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                         gridspec_kw={"width_ratios": [1.4, 1]})

ax = axes[0]
ax.plot(x, yline, color=NAVY, lw=2.6, zorder=2, label=r"model $w^\top x$")
ax.vlines(x, np.minimum(y, yline), np.maximum(y, yline),
          color=GRAY, lw=0.8, zorder=1, alpha=0.7)
ax.scatter(x, y, color=NAVY, s=20, zorder=3, label="data")
ax.set_title("Each point sits a bit off the line")
ax.set_xlabel(r"$x_i$")
ax.set_ylabel(r"$y_i$")
ax.legend(loc="upper left")

ax = axes[1]
resid = y - yline
ax.hist(resid, bins=20, density=True, color=NAVY, alpha=0.45,
        edgecolor="white", label="residuals")
grid = np.linspace(-4, 4, 200)
pdf = np.exp(-grid**2 / (2 * resid.std()**2)) / (resid.std() * np.sqrt(2*np.pi))
ax.plot(grid, pdf, color=ORANGE, lw=2.8, label="Gaussian fit")
ax.set_title("...and the errors look Gaussian")
ax.set_xlabel(r"residual  $r_i = y_i - w^\top x_i$")
ax.set_ylabel("density")
ax.legend()

fig.suptitle(r"Assumption: errors $\epsilon_i \sim \mathcal{N}(0,1)$  "
             r"$\Rightarrow$  MLE = minimize squared error",
             fontsize=14.5, fontweight="bold", y=1.01)
save(fig, "fig3_gaussian_errors.png")
