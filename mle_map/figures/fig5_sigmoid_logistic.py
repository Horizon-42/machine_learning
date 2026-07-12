"""Figure 5 - Sigmoid likelihood and the logistic loss.

Left : the linear score z = w^T x lives in (-inf, +inf); the sigmoid squashes it
       into a probability in (0,1).  That is our model p(y=+1 | x) = sigma(z).
Right: the negative log-likelihood of that sigmoid model is the logistic loss
       log(1 + exp(-y * z)).  It is a smooth, convex upper bound on the 0-1 loss
       (which just counts mistakes).  Minimizing it = MLE under the sigmoid model.
"""
import numpy as np
from _style import NAVY, ORANGE, GREEN, GRAY, setup, save
import matplotlib.pyplot as plt

setup()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# ---- sigmoid ---------------------------------------------------------------
ax = axes[0]
z = np.linspace(-6, 6, 400)
ax.plot(z, sigmoid(z), color=NAVY)
ax.axhline(1, color="#CCC", ls=":", lw=1.2)
ax.axhline(0, color="#CCC", ls=":", lw=1.2)
ax.axhline(0.5, color="#CCC", ls=":", lw=1.2)
ax.axvline(0, color="#BBB", lw=1)
ax.set_title(r"Sigmoid turns a score into a probability")
ax.set_xlabel(r"$z = w^\top x$")
ax.set_ylabel(r"$\sigma(z) = \dfrac{1}{1+e^{-z}}$")
ax.set_ylim(-0.08, 1.08)

# ---- logistic loss vs 0-1 loss (as a function of the margin y*z) ------------
ax = axes[1]
m = np.linspace(-4, 4, 400)                  # margin = y * z
zero_one = (m < 0).astype(float)
ax.plot(m, np.log(1 + np.exp(-m)), color=ORANGE,
        label=r"logistic loss $\log(1+e^{-y\,z})$")
ax.step(m, zero_one, where="post", color=GRAY, lw=2.0,
        label="0-1 loss (counts mistakes)")
ax.axvline(0, color="#BBB", lw=1)
ax.set_title("Logistic loss: a smooth, convex stand-in")
ax.set_xlabel(r"margin  $y_i\, z_i = y_i\, w^\top x_i$")
ax.set_ylabel("loss")
ax.set_ylim(-0.15, 3.2)
ax.legend()

fig.suptitle("Logistic regression = MLE with a sigmoid likelihood",
             fontsize=15, fontweight="bold", y=1.00)
save(fig, "fig5_sigmoid_logistic.png")
