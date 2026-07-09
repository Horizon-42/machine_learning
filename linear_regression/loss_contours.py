"""Contour plot of the squared loss L(w) = 0.5 * ||Xw - y||^2 for a rank-deficient X.

Because col2(X) = 3 * col1(X), the loss depends on w only through s = w1 + 3*w2,
so the contours are parallel straight lines and the minimizers form a whole line.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
X = np.array([[2.0, 6.0],
              [3.0, 9.0],
              [-1.0, -3.0]])
y = np.array([7.0, 4.0, 5.0])


def loss(w1, w2):
    """L(w) = 0.5 * ||Xw - y||^2, vectorized over grids of w1, w2."""
    w = np.stack([w1, w2], axis=-1)          # (..., 2)
    resid = w @ X.T - y                      # (..., 3)
    return 0.5 * np.sum(resid ** 2, axis=-1)


# ----------------------------------------------------------------------
# Minimizers: solve the normal equations X^T X w = X^T y
# X^T X is singular, so use the pseudoinverse -> minimum-norm solution w*
# ----------------------------------------------------------------------
w_star = np.linalg.pinv(X) @ y               # = [0.15, 0.45]
null_vec = np.array([3.0, -1.0])             # spans null(X):  X @ [3, -1] = 0
gradient_vec = np.array([1, 3])                # spans row(X), spnas(X^T):  gradient_vec @ X = [1, 3] @ X = [0, 0, 0]
c = w_star @ np.array([1.0, 3.0])            # = 1.5, so the line is w1 + 3*w2 = c

print(f"minimum-norm solution w*  = {w_star}")
print(f"solution line             : w1 + 3*w2 = {c:.4f}")
print(f"check X @ null_vec        = {X @ null_vec}")
print(f"minimum loss L(w*)        = {loss(*w_star):.4f}")

# ----------------------------------------------------------------------
# Grid + contours
# ----------------------------------------------------------------------
lim = (-5, 10)
g = np.linspace(*lim, 400)
W1, W2 = np.meshgrid(g, g)
L = loss(W1, W2)

fig, ax = plt.subplots(figsize=(7, 7))

levels = np.arange(400, L.max(), 400)        # same spacing as the exercise figure
cs = ax.contour(W1, W2, L, levels=levels, cmap="viridis", linewidths=0.8)
ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7, fmt="%d")

# ----------------------------------------------------------------------
# The line of minimizers: w(t) = w* + t * null_vec
# ----------------------------------------------------------------------
t = np.linspace(-6, 6, 2)
line = w_star[:, None] + null_vec[:, None] * t
ax.plot(line[0], line[1], color="crimson", lw=2.5, zorder=3,
        label=r"minimizers:  $w_1 + 3w_2 = \frac{3}{2}$")

w_circle = np.array([4, 4])
gradient_line = w_circle[:, None] + gradient_vec[:, None] * t
ax.plot(gradient_line[0], gradient_line[1], color="darkorange", lw=2.5, zorder=3,
        label=r"gradient line:  $w_1 + 3w_2 = 16$")

# Minimum-norm solution and two convenient points on the line
ax.plot(*w_star, marker="*", ms=16, color="crimson", mec="white", mew=0.8,
        zorder=5, ls="none", label=r"$w^\star=(0.15,\,0.45)$  (min. norm)")

print(w_star)

ax.plot(*[0, 0.5], marker=".", ms=16, color="blue", mec="white", mew=0.8,
        zorder=5, ls="none")

ax.plot(1.5, 0, marker=".", ms=16, color="blue", mec="white", mew=0.8,
        zorder=5, ls="none")

for p in [(0.0, 0.5), (6.0, -1.5)]:
    ax.plot(*p, marker="o", ms=6, mfc="white", mec="crimson", mew=1.5, zorder=4)

# A non-optimal point for contrast (the red dot in the exercise figure)
ax.plot(4, 4, marker="o", ms=7, color="darkorange", zorder=4,
        label=rf"$w=(4,4)$,  $\mathcal{{L}}={loss(4.0, 4.0):.0f}$")

# ----------------------------------------------------------------------
# Cosmetics
# ----------------------------------------------------------------------
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")                       # essential: slope -1/3 must LOOK like -1/3
ax.set_xlabel(r"$w_1$")
ax.set_ylabel(r"$w_2$")
ax.set_title(r"Contours of $\mathcal{L}(w)=\frac{1}{2}\|Xw-y\|_2^2$")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig("loss_contours.png", dpi=160)
plt.show()
