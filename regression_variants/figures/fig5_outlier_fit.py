"""Figure 5 - One outlier, two fitted lines.

Same data, but fit once by minimizing the squared error (L2) and once by
minimizing the absolute error (L1).  The squared-error line is dragged towards
the single outlier; the absolute-error line stays with the trend.
"""
import numpy as np
from scipy.optimize import minimize
from _style import NAVY, GREEN, RED, setup, save
import matplotlib.pyplot as plt

rng = np.random.RandomState(3)

# a clean downward trend ...
x = np.linspace(0, 10, 24)
y = 8 - 0.6 * x + rng.randn(24) * 0.4
# ... plus one gross outlier that does NOT follow the trend
x = np.append(x, 7.0)
y = np.append(y, 11.0)

X = np.column_stack([np.ones_like(x), x])         # design matrix [1, x]

# L2 fit = ordinary least squares (closed form)
w_l2 = np.linalg.lstsq(X, y, rcond=None)[0]

# L1 fit = minimize sum of absolute residuals (numerically)
w_l1 = minimize(lambda w: np.sum(np.abs(X @ w - y)),
                w_l2, method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x

setup()
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.scatter(x[:-1], y[:-1], color=NAVY, s=45, zorder=3, label="inliers")
ax.scatter(x[-1], y[-1], color=RED, s=110, zorder=4, marker="o",
           edgecolor="black", label="outlier")

xs = np.linspace(-0.3, 10.3, 100)
ax.plot(xs, w_l2[0] + w_l2[1] * xs, color="#C9302C", lw=2.6,
        label=r"squared error (L2) - dragged up")
ax.plot(xs, w_l1[0] + w_l1[1] * xs, color=GREEN, lw=2.6,
        label=r"absolute error (L1) - stays robust")

ax.annotate("outlier not following the trend", (7.0, 11.0),
            textcoords="offset points", xytext=(-40, -28),
            fontsize=10, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_title("Squared error chases the outlier; absolute error resists it")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.legend(loc="lower left")
save(fig, "fig5_outlier_fit.png")
