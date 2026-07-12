"""Figure 6 - The full robustness spectrum, from brittle to very robust.

Left  : four per-residual losses, ordered by how much they care about big errors:
          squared  >  absolute  >  square-root      (less and less outlier-sensitive)
        plus the L-infinity idea (only the single largest residual matters).
Right : a cartoon of convexity - squared & absolute are convex (one basin),
        the square-root error is non-convex (several basins => local minima).
"""
import numpy as np
from _style import NAVY, GREEN, PURPLE, ORANGE, RED, setup, save
import matplotlib.pyplot as plt

setup()
r = np.linspace(-3, 3, 1000)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# ---- left: the loss ordering ------------------------------------------------
ax = axes[0]
ax.plot(r, 0.5 * r**2,          color=NAVY,   label=r"squared $\frac{1}{2} r^2$  (not robust)")
ax.plot(r, np.abs(r),           color=GREEN,  label=r"absolute $|r|$  (more robust)")
ax.plot(r, np.sqrt(np.abs(r)),  color=PURPLE, label=r"square-root $\sqrt{|r|}$  (very robust)")
ax.set_title("Grows slower with $|r|$  =>  cares less about outliers")
ax.set_xlabel(r"residual  $r_i$")
ax.set_ylabel("loss")
ax.set_ylim(-0.15, 4.2)
ax.legend(loc="upper center")

# ---- right: convex vs non-convex objective (1-D slice) ----------------------
ax = axes[1]
w = np.linspace(-3, 3, 800)
# a convex bowl (squared/absolute-type) vs a wavy non-convex landscape
convex = 0.6 * (w - 0.4) ** 2 + 0.2
nonconvex = 0.9 + 0.7 * np.cos(2.3 * w) + 0.06 * (w) ** 2
ax.plot(w, convex, color=GREEN, label="convex loss (L1/L2): one global minimum")
ax.plot(w, nonconvex, color=PURPLE,
        label="non-convex loss (sqrt): many local minima")
# mark local minima of the non-convex curve
from scipy.signal import argrelextrema
mins = argrelextrema(nonconvex, np.less)[0]
ax.plot(w[mins], nonconvex[mins], "o", color=RED, ms=7, zorder=5)
ax.set_title("Price of extra robustness: non-convexity")
ax.set_xlabel(r"a weight  $w$")
ax.set_ylabel("objective")
ax.legend(loc="upper center", fontsize=10.5)

fig.suptitle("Squared -> absolute -> square-root: more robust, but harder to optimize",
             fontsize=14.5, fontweight="bold", y=1.09)
fig.subplots_adjust(top=0.86)
save(fig, "fig6_robustness_zoo.png")
