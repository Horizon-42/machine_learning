"""Figure 2 - Why we can maximize the log-likelihood instead.

log is strictly increasing, so it does not move the location of the maximum:
argmax of f  ==  argmax of log(f).  Adding a minus sign turns the maximization
into a minimization (the negative log-likelihood, NLL).  Same peak location,
easier algebra (products become sums).
"""
import numpy as np
from _style import NAVY, ORANGE, RED, PURPLE, setup, save
import matplotlib.pyplot as plt

setup()
w = np.linspace(1e-3, 1 - 1e-3, 600)
like = w**2 * (1 - w)                    # likelihood
loglike = np.log(like)                   # log-likelihood
nll = -loglike                           # negative log-likelihood
w_hat = 2 / 3
peak = w_hat**2 * (1 - w_hat)            # likelihood value at the MLE

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

ax = axes[0]
# twin axes: the likelihood and log-likelihood live on very different scales,
# so give each its own y-axis -- then it is obvious they peak at the SAME w.
l1, = ax.plot(w, like, color=NAVY, label=r"likelihood $p(\mathcal{D}\,|\,w)$")
ax.set_ylabel("likelihood", color=NAVY)
ax.tick_params(axis="y", labelcolor=NAVY)
ax.set_ylim(0, 0.17)

ax2 = ax.twinx()
ax2.grid(False)
l2, = ax2.plot(w, loglike, color=ORANGE,
               label=r"log-likelihood $\log p(\mathcal{D}\,|\,w)$")
ax2.set_ylabel("log-likelihood", color=ORANGE)
ax2.tick_params(axis="y", labelcolor=ORANGE)

ax.axvline(w_hat, color=RED, ls="--", lw=1.6, zorder=0)
ax.plot(w_hat, peak, "o", color=NAVY, ms=8, zorder=5)
ax2.plot(w_hat, np.log(peak), "o", color=ORANGE, ms=8, zorder=5)
ax.set_title("Taking log keeps the peak in the same place")
ax.set_xlabel(r"$w$")
ax.legend(handles=[l1, l2], loc="lower center")
ax.annotate(r"both peak at $\hat w = \frac{2}{3}$", (w_hat, peak),
            textcoords="offset points", xytext=(10, -18), color=RED, fontsize=11)

ax = axes[1]
ax.plot(w, nll, color=PURPLE,
        label=r"NLL $= -\log p(\mathcal{D}\,|\,w)$")
ax.axvline(w_hat, color=RED, ls="--", lw=1.6)
ax.plot(w_hat, -np.log(w_hat**2 * (1 - w_hat)), "o", color=RED, ms=9, zorder=5)
ax.set_title("Minimizing the NLL = maximizing the likelihood")
ax.set_xlabel(r"$w$")
ax.set_ylabel("negative log-likelihood")
ax.set_ylim(1.0, 6.0)
ax.legend(loc="upper center")
ax.annotate(r"$\arg\min$ at $\hat w = 2/3$", (w_hat, 1.9),
            textcoords="offset points", xytext=(10, 0), color=RED, fontsize=11)

fig.subplots_adjust(wspace=0.5)
save(fig, "fig2_log_monotonic.png")
