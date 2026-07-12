"""Figure 4 - Every loss is a negative log-likelihood in disguise.

Left : two noise models for the residual r - a Gaussian and a Laplace density.
Right: their negative logs.  -log(Gaussian) is the squared error (a parabola);
       -log(Laplace) is the absolute error (a V).  So:
         squared loss  <=> assuming Gaussian noise
         absolute loss <=> assuming Laplace noise (heavier tails -> robust)
The Laplace puts more probability on large residuals, so a big outlier is
'less surprising' and pulls the fit less -> that is why L1 is robust.
"""
import numpy as np
from _style import NAVY, GREEN, setup, save
import matplotlib.pyplot as plt

setup()
r = np.linspace(-4, 4, 800)

gauss = np.exp(-r**2 / 2) / np.sqrt(2 * np.pi)     # N(0,1)
laplace = 0.5 * np.exp(-np.abs(r))                 # Laplace(0,1)

# negative logs, shifted so both bottom out at 0 (constants don't affect argmin)
nll_gauss = r**2 / 2
nll_laplace = np.abs(r)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

ax = axes[0]
ax.plot(r, gauss, color=NAVY, label="Gaussian density")
ax.plot(r, laplace, color=GREEN, label="Laplace density (heavier tails)")
ax.fill_between(r, gauss, color=NAVY, alpha=0.06)
ax.fill_between(r, laplace, color=GREEN, alpha=0.06)
ax.set_title("Two assumptions about the noise")
ax.set_xlabel(r"residual  $r$")
ax.set_ylabel("probability density")
ax.legend()

ax = axes[1]
ax.plot(r, nll_gauss, color=NAVY, label=r"$-\log$ Gaussian $= \frac{1}{2}r^2$ (squared)")
ax.plot(r, nll_laplace, color=GREEN, label=r"$-\log$ Laplace $= |r|$ (absolute)")
ax.set_title(r"$-\log(\text{density})$  is exactly the loss")
ax.set_xlabel(r"residual  $r$")
ax.set_ylabel(r"negative log-likelihood")
ax.set_ylim(-0.2, 4.2)
ax.legend()

fig.suptitle("Choosing a loss = choosing a noise distribution",
             fontsize=15, fontweight="bold", y=1.00)
save(fig, "fig4_loss_from_likelihood.png")
