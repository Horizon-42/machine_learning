"""Figure 6 - MLE vs MAP:  posterior = likelihood x prior.

A 1-D cartoon of Bayes' rule.  The likelihood (what the data alone prefer) peaks
at w_MLE.  The prior (what we believed before seeing data) peaks at 0 and says
'small weights are more plausible'.  Their product, the posterior, peaks
somewhere in between at w_MAP - pulled towards 0.  That pull is regularization.
"""
import numpy as np
from _style import NAVY, ORANGE, GREEN, RED, setup, save
import matplotlib.pyplot as plt

setup()
w = np.linspace(-1, 6, 800)

# likelihood: data prefer w around 4 (fairly informative)
mu_like, s_like = 4.0, 0.8
likelihood = np.exp(-(w - mu_like)**2 / (2 * s_like**2))

# prior: Gaussian centred at 0 -> "keep w small"
s_prior = 1.3
prior = np.exp(-(w)**2 / (2 * s_prior**2))

# posterior (unnormalized) = likelihood * prior ; for Gaussians it is Gaussian
post = likelihood * prior
w_mle = mu_like
# closed form for product of two Gaussians' mean:
w_map = (mu_like / s_like**2 + 0.0 / s_prior**2) / (1/s_like**2 + 1/s_prior**2)

# scale each curve to the same height for readability
def norm(v):
    return v / v.max()

fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.plot(w, norm(prior), color=GREEN, lw=2.4, label=r"prior $p(w)$  (believe $w$ small)")
ax.plot(w, norm(likelihood), color=NAVY, lw=2.4, label=r"likelihood $p(\mathcal{D}\,|\,w)$")
ax.plot(w, norm(post), color=ORANGE, lw=3.0, label=r"posterior $\propto$ likelihood $\times$ prior")
ax.fill_between(w, norm(post), color=ORANGE, alpha=0.10)

for x, txt, col, dy in [
    (w_mle, r"$\hat w_{\mathrm{MLE}}$", NAVY, 0),
    (w_map, r"$\hat w_{\mathrm{MAP}}$", ORANGE, 0),
    (0.0,   "prior mean", GREEN, 0),
]:
    ax.axvline(x, color=col, ls="--", lw=1.5, alpha=0.8)
ax.annotate("", xy=(w_map, 1.08), xytext=(w_mle, 1.08),
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.text((w_map + w_mle) / 2, 1.12, "prior pulls the estimate toward 0",
        ha="center", color=RED, fontsize=11)
ax.text(w_mle, -0.09, r"$\hat w_{\mathrm{MLE}}$", ha="center", color=NAVY, fontsize=12)
ax.text(w_map, -0.09, r"$\hat w_{\mathrm{MAP}}$", ha="center", color=ORANGE, fontsize=12)

ax.set_title("MAP = MLE, but regularized by a prior")
ax.set_xlabel(r"parameter  $w$")
ax.set_ylabel("relative height")
ax.set_ylim(-0.14, 1.25)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
print("w_MLE =", round(w_mle, 3), "  w_MAP =", round(w_map, 3))
save(fig, "fig6_map_posterior.png")
