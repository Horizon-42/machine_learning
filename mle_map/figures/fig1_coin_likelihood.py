"""Figure 1 - The coin-flip likelihood.

We flipped a coin 3 times and saw H, H, T.  With w = P(heads), the likelihood of
that exact sequence is  p(HHT | w) = w * w * (1-w) = w^2 (1-w).
The Maximum Likelihood Estimate is the w that maximizes this curve: w_hat = 2/3.
Note the peak is NOT at 0.5 (we saw more H than T) and the curve is 0 at w=0,1.
"""
import numpy as np
from _style import NAVY, RED, setup, save
import matplotlib.pyplot as plt

setup()
w = np.linspace(0, 1, 500)
like = w**2 * (1 - w)                 # p(HHT | w)
w_hat = 2 / 3
peak = w_hat**2 * (1 - w_hat)

fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.plot(w, like, color=NAVY)
ax.fill_between(w, like, color=NAVY, alpha=0.08)

ax.axvline(w_hat, color=RED, ls="--", lw=1.8)
ax.plot(w_hat, peak, "o", color=RED, ms=10, zorder=5)
ax.annotate(r"MLE  $\hat{w}=\dfrac{2}{3}$", (w_hat, peak),
            textcoords="offset points", xytext=(12, -4), fontsize=13, color=RED)

ax.axvline(0.5, color="#999", ls=":", lw=1.4)
ax.annotate("fair coin\n$w=0.5$", (0.5, 0.5*0.5*0.5),
            textcoords="offset points", xytext=(-70, 30), fontsize=10,
            color="#666", ha="center",
            arrowprops=dict(arrowstyle="->", color="#999"))

ax.set_title(r"Likelihood of the data $\mathcal{D}=(H,H,T)$ as a function of $w$")
ax.set_xlabel(r"$w = P(\text{heads})$")
ax.set_ylabel(r"$p(\mathcal{D}\,|\,w) = w^2(1-w)$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.17)
save(fig, "fig1_coin_likelihood.png")
