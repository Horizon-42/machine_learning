"""
Generate all figures for the PCA tutorial.
Run: python3 make_figures.py
Figures are written to OUT as PNGs.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

OUT = "/mnt/user-data/outputs"
rng = np.random.default_rng(0)

# Semantic colors used throughout (match the tutorial text)
C_DATA = "#E24B4A"     # data points (red)
C_PROJ = "#1D9E75"     # projections (green)
C_ERR  = "#378ADD"     # reconstruction error (blue)
C_LINE = "#D4537E"     # principal subspace (magenta/pink)
C_ACC  = "#534AB7"     # accent (purple)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ----------------------------------------------------------------------
# Shared 2D dataset (centered, elongated cloud)
# ----------------------------------------------------------------------
mean = np.array([0.0, 0.0])
cov = np.array([[3.0, 1.5], [1.5, 1.2]])
X2 = rng.multivariate_normal(mean, cov, size=180)
X2 = X2 - X2.mean(axis=0)                      # ensure centered
S = np.cov(X2, rowvar=False, bias=True)        # (1/n) X^T X
evals, evecs = np.linalg.eigh(S)               # ascending
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
u1, u2 = evecs[:, 0], evecs[:, 1]              # principal, minor


def project(points, direction):
    """Orthogonal projection of each row onto a unit direction; return foot points and coords."""
    d = direction / np.linalg.norm(direction)
    coords = points @ d
    feet = np.outer(coords, d)
    return feet, coords


# ----------------------------------------------------------------------
# FIGURE 1 — projection geometry: best direction vs a poor direction
# ----------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    dirs = [(u1, "Principal direction  (max variance = min error)"),
            (u2, "Orthogonal direction  (min variance = max error)")]
    L = 6.5
    for ax, (d, title) in zip(axes, dirs):
        feet, coords = project(X2, d)
        # error segments
        for p, f in zip(X2, feet):
            ax.plot([p[0], f[0]], [p[1], f[1]], color=C_ERR, lw=0.7, alpha=0.6, zorder=1)
        # subspace line
        ax.plot([-L*d[0], L*d[0]], [-L*d[1], L*d[1]], color=C_LINE, lw=2.2, zorder=2)
        # data + projections
        ax.scatter(X2[:, 0], X2[:, 1], s=16, color=C_DATA, alpha=0.85, zorder=3, label="data $x_n$")
        ax.scatter(feet[:, 0], feet[:, 1], s=14, color=C_PROJ, zorder=4, label="projection $\\tilde{x}_n$")
        var = coords.var()
        ax.set_title(f"{title}\nprojected variance = {var:.2f}", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("Projecting the data onto a 1-D subspace", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_projection.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 2 — projected variance and reconstruction error vs angle
# ----------------------------------------------------------------------
def fig2():
    thetas = np.linspace(0, np.pi, 400)
    total = np.trace(S)
    var = np.array([np.array([np.cos(t), np.sin(t)]) @ S @ np.array([np.cos(t), np.sin(t)])
                    for t in thetas])
    err = total - var
    best = thetas[np.argmax(var)]
    worst = thetas[np.argmin(var)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    deg = np.degrees(thetas)
    ax.plot(deg, var, color=C_PROJ, lw=2.3, label="projected variance  $u^T S u$")
    ax.plot(deg, err, color=C_ERR, lw=2.3, ls="--", label="reconstruction error  (total $-$ variance)")
    ax.axhline(total, color=C_ACC, lw=1.0, ls=":", label=f"total variance = {total:.2f}")
    ax.axvline(np.degrees(best), color=C_LINE, lw=1.2, alpha=0.8)
    ax.scatter([np.degrees(best)], [var.max()], color=C_LINE, zorder=5)
    ax.annotate("max variance\n= min error\n(1st principal direction)",
                xy=(np.degrees(best), var.max()),
                xytext=(np.degrees(best) + 12, var.max() - 0.9),
                fontsize=9, color=C_LINE,
                arrowprops=dict(arrowstyle="->", color=C_LINE, lw=1))
    ax.set_xlabel("angle of the projection line  (degrees)")
    ax.set_ylabel("variance")
    ax.set_title("Variance and error are complementary: maximizing one minimizes the other")
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim(0, 180)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_variance_vs_angle.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 3 — the Lagrange picture: contours of u^T S u + unit circle + gradients
# ----------------------------------------------------------------------
def fig3():
    r = 2.4
    g = np.linspace(-r, r, 400)
    XX, YY = np.meshgrid(g, g)
    F = (S[0, 0]*XX**2 + 2*S[0, 1]*XX*YY + S[1, 1]*YY**2)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    cs = ax.contour(XX, YY, F, levels=12, cmap="Greys", linewidths=0.9, alpha=0.9)
    # unit circle (the constraint)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=C_LINE, lw=2.3, zorder=3))
    # eigen-directions where the circle is tangent to a contour
    for d, lam, name in [(u1, evals[0], "$u_1$ (max)"), (u2, evals[1], "$u_2$ (min)")]:
        p = d if d[0] >= 0 else -d
        ax.scatter(*p, color=C_ACC, s=55, zorder=5)
        ax.annotate(name, xy=p, xytext=p*1.35, fontsize=11, color=C_ACC,
                    ha="center", va="center")
        # gradient of f (blue) and gradient of g (gray) at the max point -> parallel
        if name.startswith("$u_1$"):
            gf = 2 * S @ p          # ∇f = 2 S u
            gg = 2 * p              # ∇g = 2 u
            gf = gf / np.linalg.norm(gf) * 0.9
            gg = gg / np.linalg.norm(gg) * 0.9
            ax.annotate("", xy=p+gf, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=C_ERR, lw=2.2))
            ax.annotate("", xy=p+gg, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color="#888780", lw=2.2))
            ax.text(*(p+gf*1.05+np.array([0.05, 0.05])), r"$\nabla f$", color=C_ERR, fontsize=11)
            ax.text(*(p+gg*1.05+np.array([0.08,-0.18])), r"$\nabla g$", color="#5F5E5A", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.set_title("Constraint circle $u^Tu=1$ (pink) over contours of $u^TSu$.\n"
                 "At the optimum the gradients align: $Su=\\lambda u$.", fontsize=10.5)
    ax.set_xlabel("$u_x$"); ax.set_ylabel("$u_y$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_lagrange.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Digits data for figures 4 and 5
# ----------------------------------------------------------------------
digits = load_digits()
Xd = digits.data.astype(float)            # (1797, 64)
Xd_mean = Xd.mean(axis=0)
Xc = Xd - Xd_mean
# SVD-based PCA
U, sig, Vt = np.linalg.svd(Xc, full_matrices=False)
evr = sig**2 / np.sum(sig**2)             # explained variance ratio
cum = np.cumsum(evr)


# ----------------------------------------------------------------------
# FIGURE 4 — scree plot + cumulative variance explained
# ----------------------------------------------------------------------
def fig4():
    k_show = 32
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    idx = np.arange(1, k_show + 1)
    ax.bar(idx, evr[:k_show], color=C_ACC, alpha=0.75, label="variance explained by each component")
    ax.set_xlabel("principal component index")
    ax.set_ylabel("explained variance ratio", color=C_ACC)
    ax.tick_params(axis="y", labelcolor=C_ACC)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(idx, cum[:k_show], color=C_PROJ, lw=2.3, marker="o", ms=3,
             label="cumulative variance explained")
    ax2.axhline(0.9, color=C_ERR, ls="--", lw=1)
    k90 = int(np.argmax(cum >= 0.9) + 1)
    ax2.annotate(f"90% at k={k90}", xy=(k90, 0.9), xytext=(k90+2, 0.72),
                 fontsize=9, color=C_ERR,
                 arrowprops=dict(arrowstyle="->", color=C_ERR))
    ax2.set_ylabel("cumulative", color=C_PROJ)
    ax2.tick_params(axis="y", labelcolor=C_PROJ)
    ax2.set_ylim(0, 1.02)
    ax.set_title("Scree plot: how variance is distributed across components (digits, d=64)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_scree.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 5 — reconstruction of one digit at increasing k
# ----------------------------------------------------------------------
def fig5():
    sample = 12  # a digit sample index
    ks = [1, 4, 8, 16, 32, 64]
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(11.5, 2.4))
    axes[0].imshow(Xd[sample].reshape(8, 8), cmap="gray_r")
    axes[0].set_title("original", fontsize=10)
    axes[0].axis("off")
    for ax, k in zip(axes[1:], ks):
        Wk = Vt[:k]                       # (k, 64) principal axes
        z = Xc[sample] @ Wk.T             # (k,) code
        recon = z @ Wk + Xd_mean          # back to pixel space
        ax.imshow(recon.reshape(8, 8), cmap="gray_r")
        ax.set_title(f"k={k}\n{cum[k-1]*100:.0f}% var", fontsize=9.5)
        ax.axis("off")
    fig.suptitle("Reconstruction improves as k grows — variance explained is reconstruction quality",
                 fontsize=12, y=1.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_reconstruction.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 6 — PCA (soft blend) vs VQ / k-means (hard assignment)
# ----------------------------------------------------------------------
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # PCA reconstruction with k=1: every point lands on the principal line
    feet, _ = project(X2, u1)
    ax = axes[0]
    L = 6.5
    ax.plot([-L*u1[0], L*u1[0]], [-L*u1[1], L*u1[1]], color=C_LINE, lw=2.0, zorder=2)
    ax.scatter(X2[:, 0], X2[:, 1], s=16, color=C_DATA, alpha=0.4, zorder=3, label="original")
    ax.scatter(feet[:, 0], feet[:, 1], s=15, color=C_PROJ, zorder=4, label="reconstruction")
    ax.set_title("PCA ($k=1$): soft blend\neach point $\\approx z_i \\cdot u_1$  (a real-valued weight)", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.legend(loc="upper left", fontsize=8.5)

    # VQ / k-means with k=4: every point snaps to its nearest centroid
    ax = axes[1]
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X2)
    cen = km.cluster_centers_
    lab = km.labels_
    recon = cen[lab]
    palette = [C_ACC, C_PROJ, C_ERR, "#BA7517"]
    for j in range(4):
        m = lab == j
        ax.scatter(X2[m, 0], X2[m, 1], s=16, color=palette[j], alpha=0.35, zorder=3)
    ax.scatter(recon[:, 0], recon[:, 1], s=10, color="#555", alpha=0.15, zorder=2)
    ax.scatter(cen[:, 0], cen[:, 1], s=180, marker="*", color="k", zorder=5, label="centroids $w_j$")
    ax.set_title("VQ / k-means ($k=4$): hard assignment\neach point $\\to w_{z_i}$  (one integer index)", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Same $ZW$ skeleton, opposite codes: continuous weights vs a single index", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_pca_vs_vq.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("wrote figures to", OUT)
    print("eigenvalues (2D):", np.round(evals, 3))
    print("digits: 90% var at k =", int(np.argmax(cum >= 0.9) + 1),
          "| 95% at k =", int(np.argmax(cum >= 0.95) + 1))
