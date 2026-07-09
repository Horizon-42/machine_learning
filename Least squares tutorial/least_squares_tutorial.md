# Least-Squares: unique vs. infinitely many solutions

A practical guide to the family of exam questions built around

$$\mathcal{L}(w) = \tfrac{1}{2}\lVert Xw - y\rVert_2^2 .$$

These questions all probe the same three things: **how many minimizers exist**, **which one is special**, and **which one gradient descent actually finds**. Once you see that all three are controlled by a single property of $X$ — the rank of its columns — the whole topic collapses into one short checklist.

---

## 0. The objects you always start from

For a dataset with $n$ samples and $d$ features, $X$ is $n \times d$, $y$ is $n \times 1$, and $w$ is $d \times 1$.

| Object | Formula | Meaning |
|---|---|---|
| Loss | $\mathcal{L}(w) = \tfrac12\lVert Xw - y\rVert^2$ | squared error, a convex quadratic in $w$ |
| Gradient | $\nabla\mathcal{L}(w) = X^\top(Xw - y)$ | zero at every minimizer |
| Normal equations | $X^\top X\, w = X^\top y$ | the condition $\nabla\mathcal{L}=0$, rewritten |

Two facts that never change, whatever $X$ is:

- $\mathcal{L}$ is **convex**, so any stationary point is a global minimum — there are no local traps.
- A minimizer **always exists** (it is the orthogonal projection of $y$ onto the column space of $X$). The only open question is whether it is *unique*.

---

## 1. The one question that decides everything

> **Does $X$ have full column rank?** That is, are its $d$ columns linearly independent?

Everything downstream branches on the answer.

| | Full column rank, $\operatorname{rank}(X)=d$ | Rank-deficient, $\operatorname{rank}(X)<d$ |
|---|---|---|
| $X^\top X$ | invertible ($\det \neq 0$) | singular ($\det = 0$) |
| Null space of $X$ | only $\{0\}$ | a nonzero subspace |
| Minimizers | **exactly one** | **infinitely many** (an affine subspace) |
| Contours of $\mathcal{L}$ | closed ellipses | parallel flat lines/troughs |
| Gradient descent | converges to *the* solution | converges to a solution that depends on the start |

### How to check rank quickly

1. **Eyeball for dependence.** Is one column a scalar multiple of another (or a combination of others)? If yes → rank-deficient.
2. **Compute $\det(X^\top X)$.** Zero → rank-deficient. (Or ask for `numpy.linalg.matrix_rank(X)`.)
3. **Dimension sanity check.** If $n < d$ (fewer samples than features), the columns *cannot* all be independent → automatically rank-deficient.

---

## 2. Case A — Full column rank (the "normal" case)

**Example.** $\displaystyle X = \begin{bmatrix}1 & 0\\ 1 & 1\\ 1 & 2\end{bmatrix},\quad y = \begin{bmatrix}3\\ 1\\ 5\end{bmatrix}.$

The columns $(1,1,1)^\top$ and $(0,1,2)^\top$ are not multiples of each other, so rank $=2=d$. Confirm via

$$X^\top X = \begin{bmatrix}3 & 3\\ 3 & 5\end{bmatrix}, \qquad \det(X^\top X) = 15 - 9 = 6 \neq 0.$$

Invertible, so **solve the normal equations directly** for the unique answer:

$$w^\star = (X^\top X)^{-1}X^\top y = \frac{1}{6}\begin{bmatrix}5 & -3\\ -3 & 3\end{bmatrix}\begin{bmatrix}9\\ 11\end{bmatrix} = \begin{bmatrix}2\\ 1\end{bmatrix}.$$

The minimum loss is $\mathcal{L}(w^\star) = 3$ (the fit is not perfect; the residual is $(1,-2,1)^\top$, orthogonal to both columns).

Because there is one minimum, the contours are **closed ellipses** and gradient descent spirals into $(2,1)$ from *any* starting point:

![Full-rank contours: closed ellipses with a single minimum, two gradient-descent paths from opposite corners both converging to (2,1)](fullrank_contours.png)

*Takeaway for Case A:* nothing subtle happens. Solve $X^\top X w = X^\top y$, report the one answer, done.

---

## 3. Case B — Rank-deficient (infinitely many solutions)

**Example (the one we studied).** $\displaystyle X = \begin{bmatrix}+2 & +6\\ +3 & +9\\ -1 & -3\end{bmatrix},\quad y = \begin{bmatrix}+7\\ +4\\ +5\end{bmatrix}.$

### Step 1 — Diagnose the dependence

Column 2 is exactly $3\times$ column 1, so $\operatorname{rank}(X)=1<2$. Equivalently

$$X^\top X = \begin{bmatrix}14 & 42\\ 42 & 126\end{bmatrix}, \qquad \det(X^\top X) = 0.$$

### Step 2 — Find the two special directions

Whenever $\text{col}_2 = k\,\text{col}_1$, the two directions in $w$-space are:

| direction (column vector) | name | role |
|---|---|---|
| $u = \begin{bmatrix}1\\ k\end{bmatrix} = \begin{bmatrix}1\\ 3\end{bmatrix}$ | **row-space / normal** | loss changes along this; every gradient is a multiple of it |
| $v = \begin{bmatrix}k\\ -1\end{bmatrix} = \begin{bmatrix}3\\ -1\end{bmatrix}$ | **null space** | $Xv = 0$; loss is *flat* along this |

They are orthogonal: $u^\top v = k - k = 0$. This orthogonality is the backbone of everything that follows.

### Step 3 — Write the solution set

The loss sees $w$ only through the scalar $s = u^\top w = w_1 + 3w_2$. Setting $\nabla\mathcal{L}=0$ gives one equation (both normal equations are multiples of it):

$$14w_1 + 42w_2 = 21 \;\;\Longleftrightarrow\;\; \boxed{w_1 + 3w_2 = \tfrac{3}{2}}.$$

That single line **is** the set of minimizers. Two equivalent descriptions:

$$\underbrace{\{w : u^\top w = \tfrac32\}}_{\text{implicit}} = \underbrace{\{w^\star + t\,v : t\in\mathbb{R}\}}_{\text{parametric}}.$$

### Step 4 — The minimum-norm solution $w^\star$

Among the infinitely many minimizers, the **shortest** one is singled out by the pseudoinverse (or: it is the minimizer with zero null-component):

$$w^\star = X^{+}y = \tfrac{3}{20}\begin{bmatrix}1\\ 3\end{bmatrix} = \begin{bmatrix}0.15\\ 0.45\end{bmatrix}, \qquad \mathcal{L}(w^\star) = 29.25.$$

Because the loss is flat along $v$, the contours are **parallel straight lines**, not ellipses — the tell-tale sign of rank deficiency:

![Rank-deficient contours: parallel straight lines, with the line of minimizers and the minimum-norm solution marked](loss_contours.png)

*Why infinitely many?* Adding any multiple of $v$ leaves $Xw$ unchanged ($Xv=0$), hence leaves the loss unchanged. So if one minimizer exists, the whole line $w^\star + \operatorname{span}\{v\}$ minimizes.

---

## 4. Which solution does gradient descent find?

This is the part that trips people up, and it *only matters in Case B* (in Case A there is nothing to choose). The key fact:

> **Gradient descent converges to the minimizer closest to its initialization — not to $w^\star$.**

### Why: descent can only move in one direction

Every gradient is a multiple of $u$:

$$\nabla\mathcal{L}(w) = \underbrace{14\left(s - \tfrac32\right)}_{\text{scalar}}\,\underbrace{\begin{bmatrix}1\\ 3\end{bmatrix}}_{u}.$$

So every update $w^{(k+1)} = w^{(k)} - \eta\,\nabla\mathcal{L}(w^{(k)})$ steps **along $u$ and never along $v$**. Consequences:

- The iterates are trapped on the line $w^{(0)} + \operatorname{span}\{u\}$.
- The component of $w$ along the flat direction $v$ is **frozen** for all $k$.

### The projection picture

Since the trapped line runs along $u$, and $u \perp v$ (the line of minimizers runs along $v$), the descent path is exactly the **perpendicular dropped from $w^{(0)}$ onto the line of minimizers**. Where it hits is the **orthogonal projection** of the point $w^{(0)}$ onto that line:

![Projecting the point w0 = (4,4) perpendicularly onto the line of minimizers, landing at (2.55, -0.35), with a right-angle marker at the foot](projection.png)

The two defining properties of that foot — *lies on the line* and *the drop is perpendicular to the line* — are all you ever need to check.

### Computing the landing point two ways

**Way 1 — foot of the perpendicular.** Move from $w^{(0)}$ along $u$ just far enough to hit the line $u^\top w = c$:

$$w^{(\infty)} = w^{(0)} - \frac{u^\top w^{(0)} - c}{u^\top u}\,u = \begin{bmatrix}4\\4\end{bmatrix} - \frac{16 - 1.5}{10}\begin{bmatrix}1\\3\end{bmatrix} = \begin{bmatrix}2.55\\ -0.35\end{bmatrix}.$$

**Way 2 — conserved null-component.** The frozen quantity is $P_{\mathcal{N}}w$, where $P_{\mathcal{N}} = \dfrac{vv^\top}{v^\top v}$ projects onto the null direction. The limit keeps $w^{(0)}$'s null-component, and $w^\star$'s null-component is zero, so

$$w^{(\infty)} = w^\star + P_{\mathcal{N}}w^{(0)} = \begin{bmatrix}0.15\\0.45\end{bmatrix} + \begin{bmatrix}2.4\\-0.8\end{bmatrix} = \begin{bmatrix}2.55\\-0.35\end{bmatrix}.\;\checkmark$$

Both give the same point. Way 2 makes the mechanism visible: the null-vector $P_{\mathcal{N}}w^{(0)} = (2.4,-0.8)$ appears **twice** — once as a component of $w^{(0)}$ from the origin, and again as the shift that slides $w^\star$ along the line to the landing point:

![Decomposition of w0 into row-space part (1.6,4.8) and null-space part (2.4,-0.8); the same null vector translates w* along the line to the landing point](null_component.png)

### When *does* gradient descent reach $w^\star$?

Only when $w^{(0)}$ has zero null-component to begin with — i.e. $w^{(0)}$ already lies in the row space of $X$. The cleanest such choice is $w^{(0)} = 0$. **Initializing at the origin gives the minimum-norm solution.** Any other start smuggles in a null-component that no gradient step can remove.

---

## 5. The general recipe (memorize this)

```
1. Write the gradient:      ∇L = Xᵀ(Xw − y);   normal eqns  XᵀX w = Xᵀy
2. Check the rank of X:
      • det(XᵀX) ≠ 0  →  FULL RANK  → go to 3A
      • det(XᵀX) = 0  →  RANK-DEFICIENT → go to 3B

3A. FULL RANK (unique):
      w* = (XᵀX)⁻¹ Xᵀy.   Done. GD converges here from anywhere.

3B. RANK-DEFICIENT (infinitely many):
      • Null direction v: solve Xv = 0.  Loss is flat along v.
      • Solution set = { w : normal-eqn line } = { w* + t·v }.
      • Min-norm solution:  w* = X⁺y  (pseudoinverse; zero null-component).
      • GD from w⁽⁰⁾ converges to the PROJECTION of w⁽⁰⁾ onto the solution set:
              w^∞ = w⁽⁰⁾ − ((uᵀw⁽⁰⁾ − c)/(uᵀu))·u
                  = w* + P_N w⁽⁰⁾,     P_N = v vᵀ / (vᵀv)
      • GD reaches w* itself ⇔ w⁽⁰⁾ is in row(X) (e.g. w⁽⁰⁾ = 0).
```

### Trigger phrases and how to answer them

| Question asks… | What to say |
|---|---|
| "Why are there infinitely many minimizers?" | Columns of $X$ are dependent ⇒ $X$ has a nonzero null space $\operatorname{span}\{v\}$; adding any $t\,v$ leaves $Xw$ (hence the loss) unchanged. |
| "Why are the contours straight lines?" | Loss depends on $w$ only through $s = u^\top w$; it is constant along the flat direction $v$. |
| "Does GD converge to $w^\star$?" | Only if the initialization has zero null-component (lies in row space). Otherwise no — GD freezes the null-component of $w^{(0)}$. |
| "What does GD converge to?" | The orthogonal projection of $w^{(0)}$ onto the solution line: $w^{(\infty)} = w^\star + P_{\mathcal{N}}w^{(0)}$. |
| "How to draw the solution set?" | Straight line parallel to every contour, through $w^\star$; equation $u^\top w = c$. |

---

## 6. Two ideas worth carrying beyond this problem

- **Implicit bias.** Gradient descent doesn't pick a solution neutrally; it prefers the one nearest where it started. Starting at $0$ biases it toward the *smallest-norm* solution — a mild, automatic regularization that shows up all over machine learning.
- **Split the space with two projectors.** $P_u$ (onto $\operatorname{range}(X^\top)$) captures the part descent *moves*; $P_{\mathcal{N}}$ (onto $\ker X$) captures the part descent *preserves*. They satisfy $P_u + P_{\mathcal{N}} = I$. Deciding "what changes vs. what is conserved" is the fastest route through almost every variant of this question.

---

## Appendix — reproduce every figure

Each figure above was generated by a short, self-contained Python script (numpy + matplotlib):

| Figure | Script | What it shows |
|---|---|---|
| Full-rank ellipses | `fullrank_contours.py` | unique minimum, two GD paths converging |
| Rank-deficient lines | `loss_contours.py` | parallel contours, solution line, $w^\star$ |
| Projection | `projection.py` | point-to-line perpendicular drop |
| Null decomposition | `null_component.py` | $w^{(0)} = P_u w^{(0)} + P_{\mathcal{N}}w^{(0)}$, and why $P_{\mathcal{N}}$ fixes the limit |

A reliable habit from this exercise: **when a plot seems to contradict your algebra, substitute the point back into the equation before doubting the algebra.** Rendering has finite resolution; the equation does not.
