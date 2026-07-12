"""Figure 7 - The prior IS the regularizer (bridge back to Tutorial 1).

A regularizer is just a negative log-prior.  Two priors on each weight:
  * Gaussian prior  w ~ N(0, 1/lambda)   -> -log p(w) = (lambda/2) w^2  = L2 penalty
  * Laplace prior   w ~ Laplace(0, ...)  -> -log p(w) = lambda |w|      = L1 penalty
Top row: the prior densities.  Bottom row: their negative logs, i.e. the penalty
you add to the loss.  Same story as the loss<->noise link, one level up.
"""
import numpy as np
from _style import ORANGE, GREEN, setup, save
import matplotlib.pyplot as plt

setup()
w = np.linspace(-3, 3, 800)
lam = 1.0

gauss_prior = np.exp(-lam * w**2 / 2)
laplace_prior = np.exp(-lam * np.abs(w))

l2_pen = lam * w**2 / 2          # -log Gaussian prior (up to const)
l1_pen = lam * np.abs(w)         # -log Laplace prior  (up to const)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(w, gauss_prior, color=ORANGE)
axes[0, 0].fill_between(w, gauss_prior, color=ORANGE, alpha=0.08)
axes[0, 0].set_title(r"Gaussian prior  $p(w_j)\propto e^{-\frac{\lambda}{2}w_j^2}$")
axes[0, 0].set_ylabel("prior density")

axes[0, 1].plot(w, laplace_prior, color=GREEN)
axes[0, 1].fill_between(w, laplace_prior, color=GREEN, alpha=0.08)
axes[0, 1].set_title(r"Laplace prior  $p(w_j)\propto e^{-\lambda|w_j|}$")

axes[1, 0].plot(w, l2_pen, color=ORANGE)
axes[1, 0].set_title(r"$-\log$ prior $= \frac{\lambda}{2}w_j^2$   ($L_2$ penalty)")
axes[1, 0].set_xlabel(r"$w_j$")
axes[1, 0].set_ylabel("penalty added to loss")
axes[1, 0].set_ylim(-0.2, 4)

axes[1, 1].plot(w, l1_pen, color=GREEN)
axes[1, 1].set_title(r"$-\log$ prior $= \lambda|w_j|$   ($L_1$ penalty)")
axes[1, 1].set_xlabel(r"$w_j$")
axes[1, 1].set_ylim(-0.2, 4)

fig.suptitle("A prior on the weights is exactly a regularizer on the weights",
             fontsize=15, fontweight="bold", y=0.98)
save(fig, "fig7_prior_as_regularizer.png")
