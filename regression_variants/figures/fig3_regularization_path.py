"""Figure 3 - Regularization paths for L2 (ridge) vs L1 (LASSO).

Each curve is one weight w_j as the regularization strength lambda grows
(x-axis = log2(lambda), matching the slide).  Key contrast:
  * L2 shrinks every weight smoothly towards 0 but (almost) never *to* 0.
  * L1 drives weights to EXACTLY 0 one after another -> automatic feature
    selection.  Once a curve hits 0 it stays there.
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from _style import PALETTE, setup, save
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

# ---- synthetic data: a few informative features, several useless ones -------
n, d = 120, 10
X = rng.randn(n, d)
true_w = np.array([9.0, -3.5, 2.5, 0, 0, 0, 0, 0, 0, 0])  # only 3 matter
y = X @ true_w + 0.5 * rng.randn(n)

log2_lam = np.linspace(0, 14, 60)
lams = 2.0 ** log2_lam

ridge_paths = np.array([Ridge(alpha=l, fit_intercept=False).fit(X, y).coef_
                        for l in lams])
lasso_paths = np.array([Lasso(alpha=l / n, fit_intercept=False, max_iter=50000)
                        .fit(X, y).coef_ for l in lams])

setup()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

for ax, paths, title in [
    (axes[0], ridge_paths, r"$L_2$-regularization (ridge)"),
    (axes[1], lasso_paths, r"$L_1$-regularization (LASSO)"),
]:
    for j in range(d):
        ax.plot(log2_lam, paths[:, j], color=PALETTE[j % len(PALETTE)],
                lw=2.0 if abs(true_w[j]) > 0 else 1.3,
                alpha=1.0 if abs(true_w[j]) > 0 else 0.7)
    ax.axhline(0, color="#555", lw=1.0, ls="--")
    ax.set_title(title)
    ax.set_xlabel(r"$\log_2(\lambda)$  (more regularization $\rightarrow$)")
    ax.set_xlim(log2_lam.min(), log2_lam.max())

axes[0].set_ylabel(r"weight value  $w_j$")
axes[0].text(0.5, 0.05, "weights shrink but stay non-zero",
             transform=axes[0].transAxes, ha="center", fontsize=11,
             style="italic", color="#444")
axes[1].text(0.5, 0.05, "weights snap to exactly 0",
             transform=axes[1].transAxes, ha="center", fontsize=11,
             style="italic", color="#444")
fig.suptitle("Regularization path: how each weight changes with " + r"$\lambda$",
             fontsize=15, fontweight="bold", y=1.00)
save(fig, "fig3_regularization_path.png")
