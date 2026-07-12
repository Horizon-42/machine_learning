# Regression Variants — A Friendly Tutorial

*Based on the lecture "Machine Learning — Regression Variants & MLE/MAP" (A. Lehrmann, THWS). This tutorial keeps the slides' logic but fills in the math and the "why" so you can learn it from scratch.*

> **About the figures.** Every picture in this tutorial is produced by an accurate Python script in the `figures/` folder, and the rendered `.png` files are shipped alongside. Images are **not** embedded in this file — when the text says *see Figure 3*, open `figures/fig3_regularization_path.png` (or run `python3 figures/fig3_regularization_path.py` to regenerate it).

---

## 0. The one-sentence map of this tutorial

We already know how to fit a line by minimizing the **squared error**. This tutorial collects the useful **variants** of that idea:

- change the **regularizer** (what we add to keep the model simple): $L_2$, $L_0$, and the star of the show, $L_1$;
- change the **loss** (how we measure the training error): squared, absolute, Huber, square-root, $L_\infty$.

Everything is the same recipe with a different penalty or a different error measure:

$$
\mathcal{L}(w) \;=\; \underbrace{\text{data-fit term}}_{\text{loss}} \;+\; \underbrace{\lambda\cdot\text{penalty}(w)}_{\text{regularizer}} .
$$

The parameter $\lambda > 0$ is a **knob** that balances the two: small $\lambda$ = "fit the data hard", large $\lambda$ = "keep the model simple".

---

## 1. Quick recap: where we start

From the previous lecture, two facts set the stage.

**(a) Regularized least squares.** If we penalize large weights with the squared ($L_2$) norm, we minimize

$$
\mathcal{L}(w) \;=\; \tfrac{1}{2}\lVert Xw - y\rVert^2 \;+\; \tfrac{\lambda}{2}\lVert w\rVert^2 ,
$$

and this has a clean **closed-form** solution (just set the gradient to zero):

$$
w \;=\; (X^\top X + \lambda I)^{-1} X^\top y .
$$

Here $X \in \mathbb{R}^{n\times d}$ stacks the $n$ samples as rows, $y\in\mathbb{R}^n$ are the targets, and $w\in\mathbb{R}^d$ are the weights. This objective is **convex and smooth**, which is why it is so easy to solve.

**(b) Feature selection.** Sometimes we don't just want small weights — we want *most weights to be exactly zero*, so the model uses only a handful of features. The "honest" way to ask for that is the $L_0$ penalty,

$$
\lVert w\rVert_0 \;=\; \#\{\, j : w_j \neq 0 \,\} \quad(\text{the number of non-zero weights}),
$$

but minimizing it means searching over subsets of features (forward selection etc.), which is **non-convex and slow**.

That tension — $L_2$ is easy but keeps everything, $L_0$ selects but is intractable — is exactly what motivates $L_1$.

---

## 2. Feature selection by regularization: the $L_1$ norm (LASSO)

### 2.1 Why we need it

Forward selection tries adding features one at a time and re-fitting. If you have $d$ candidate features, that is roughly $\mathcal{O}(d^2)$ model fits — and each fit can itself be an iterative optimization if the loss isn't a plain squared error. With **millions** of features that is hopeless.

We want a method that does two things **at the same time and in one convex optimization**:

1. shrink weights (like $L_2$), and
2. set some weights to *exactly* zero (like $L_0$).

### 2.2 The idea

Regularize with the $L_1$ norm instead of $L_0$ or $L_2$:

$$
\boxed{\;\mathcal{L}(w) \;=\; \tfrac{1}{2}\lVert Xw - y\rVert^2 \;+\; \lambda\,\lVert w\rVert_1\;}
\qquad
\lVert w\rVert_1 = \sum_{j=1}^{d} |w_j| .
$$

This is called **LASSO** (Least Absolute Shrinkage and Selection Operator). It has the best of both worlds:

- **Like $L_2$:** it is **convex** (the absolute value is convex, and a sum of convex functions is convex), so there are no bad local minima and it reliably lowers the test error by controlling overfitting.
- **Like $L_0$:** it pushes many weights to be **exactly zero** — so it selects features for you.

That combination is why $L_1$ is often described as a *"very fast alternative to search-and-score"* feature selection.

### 2.3 The key question: *why* does $L_1$ zero things out but $L_2$ doesn't?

Both $L_1$ and $L_2$ shrink weights toward zero, so students always ask: *don't they do the same thing?* No — and the cleanest way to see the difference is to ask what penalty each one charges for a **tiny** weight, say $w_j = 0.00001$. (See **Figure 1**, `figures/fig1_norm_penalties.png`.)

| Regularizer | Penalty on one weight $w_j$ | Penalty at $w_j = 0.00001$ | Behaviour near zero |
|---|---|---|---|
| $L_0$ | $\lambda \cdot \mathbb{1}[w_j \neq 0]$ | $\lambda$ (full price!) | constant — a flat fee for being non-zero |
| $L_2$ | $\dfrac{\lambda}{2}\, w_j^2$ | $\dfrac{\lambda}{2}(10^{-5})^2 = 5\times10^{-11}\,\lambda$ | **vanishes** — almost free to keep it non-zero |
| $L_1$ | $\lambda\,|w_j|$ | $10^{-5}\,\lambda$ | **stays proportional** — still worth killing |

Read the last column carefully, because it is the whole story:

- **$L_2$** charges $\frac{\lambda}{2}w_j^2$. Near zero this is *quadratically* small, so its **slope** at $w_j=0$ is also zero. Once a weight is already tiny, $L_2$ has essentially **no incentive** to push it the last little bit to *exactly* zero — there's nothing more to gain. Result: $L_2$ leaves you with lots of small-but-non-zero weights.
- **$L_1$** charges $\lambda|w_j|$. Its slope is $\pm\lambda$ **right up to zero** — the penalty keeps rewarding you at a constant rate for shrinking, all the way down. So there is *always something to gain* by setting a tiny weight to exactly $0$. Result: weights snap to zero.
- **$L_0$** charges the full $\lambda$ for any non-zero value, no matter how small. It is the "purest" selector but is a step function — non-convex and non-differentiable — hence intractable.

**A slightly more formal version (optional but worth it).** Think about a single weight $w_j$ and freeze the others. The objective looks like $\frac{1}{2}a(w_j - c)^2 + \lambda\,\text{pen}(w_j)$ for some numbers $a>0, c$ coming from the data. Optimality means the derivative is zero.

- For $L_2$, $\text{pen}(w_j)=\frac12 w_j^2$ is differentiable everywhere, so the optimum satisfies $a(w_j-c) + \lambda w_j = 0 \Rightarrow w_j = \frac{a c}{a+\lambda}$. This is zero **only if $c=0$ exactly** — which never happens with real data.
- For $L_1$, $\text{pen}(w_j)=|w_j|$ has a **kink** at $0$. Its "derivative" there is not a single number but the whole interval $[-\lambda, +\lambda]$ (the *subgradient*). Zero is optimal whenever the data's pull $|ac|$ is **smaller than $\lambda$** — i.e. whenever the feature isn't useful enough to overcome the fixed $\lambda$ incentive. This gives a genuine *threshold*: features below it are switched off completely. (The resulting update rule is called **soft-thresholding**.)

### 2.4 The geometric picture (the classic diamond vs. circle)

There's a beautiful visual for the same fact. Instead of "loss + $\lambda\cdot$penalty", think of it as **"minimize the loss subject to a budget on the weights"** — the two views are equivalent. The budget region is:

- a **diamond** $|w_1| + |w_2| \le t$ for $L_1$, and
- a **disk** $w_1^2 + w_2^2 \le t^2$ for $L_2$.

The squared training error forms elliptical contours around the unconstrained least-squares optimum. The regularized solution is the point of the budget region that the growing ellipse **first touches**. (See **Figure 2**, `figures/fig2_sparsity_geometry.png`.)

- The **diamond has pointy corners that sit on the axes**. Ellipses tend to touch a diamond *at a corner*, and a corner means one coordinate is exactly zero → **sparsity**.
- The **disk is smooth**, so the ellipse touches it at a generic point where *both* coordinates are non-zero.

In the figure the $L_1$ solution lands exactly at $(1.5, 0)$ — $w_2$ has been switched off — while the $L_2$ solution keeps both weights alive.

### 2.5 The regularization path

A great way to *see* the difference is to sweep $\lambda$ from small to large and plot how every weight $w_j$ changes. This curve is called the **regularization path**. (See **Figure 3**, `figures/fig3_regularization_path.png`.)

- **$L_2$ path:** every weight glides **smoothly** toward zero as $\lambda$ grows, but they only reach zero in the limit $\lambda\to\infty$. At any finite $\lambda$, all weights are non-zero.
- **$L_1$ path:** weights hit **exactly zero one after another** and then *stay* there. At a given $\lambda$ only a subset of features survive — that subset is your selected feature set.

### 2.6 $L_1$ vs. $L_2$ — the cheat-sheet table

| | $L_2$ (ridge) | $L_1$ (LASSO) |
|---|---|---|
| Robust to small data changes? | yes | yes |
| Reduces variance → lower test error? | yes | yes |
| Solution method | **closed form** $(X^\top X+\lambda I)^{-1}X^\top y$ | **iterative** (e.g. coordinate descent) |
| Solution unique? | yes | **not always** (e.g. $\lVert(1,1)^\top\rVert_1 = \lVert(0,2)^\top\rVert_1 = 2$) |
| Typical weights | all non-zero | many exactly zero |
| Tolerates irrelevant features | a *linear* number, $\mathcal{O}(d)$ relevant | an *exponential* number, only $\mathcal{O}(\log d)$ relevant¹ |

¹ This last row is a real theoretical result (Ng, 2004, "Feature selection, $L_1$ vs. $L_2$ regularization, and rotational invariance"): with $L_1$ you can still learn well even when the number of irrelevant features grows exponentially, needing only about $\log d$ truly relevant ones. That is a big deal for high-dimensional problems.

---

## 3. Putting the pieces together: mix and match

Here is the payoff for keeping the loss and the regularizer as separate Lego bricks. You can **combine any loss with any regularizer, and optionally kernelize**. Many famous models are just particular cells in this table:

| | loss = squared | loss = logistic | loss = hinge |
|---|---|---|---|
| linear, no reg. | linear regression | logistic regression | hinge classifier |
| linear, $L_2$ reg. | ridge regression | $L_2$-regularized logistic regression | **support vector machine (SVM)** |
| kernelized, $L_2$ reg. | kernel ridge regression | kernel logistic regression | **kernel SVM** |

The lecture's exercise recipe for building any of these (illustrated with the kernel SVM) is worth remembering as a *general* pattern:

1. Write the loss and **replace the linear model $w^\top x_i$ by a function $f(x_i)$** in the feature space $\mathcal{F}$.
2. Plug in the **representer theorem** so that $f(\cdot)=\sum_i \alpha_i k(\cdot, x_i)$, turning the problem into one over the coefficient vector $\alpha$.
3. Compute the gradient **with respect to $\alpha$** (vectorize it — the elementwise/Hadamard product $v\odot v'$ shows up a lot).
4. Feed that gradient into a **gradient-descent** update.

You don't need to memorize the SVM algebra; the point is that *loss + regularizer + (optional) kernel* is a modular framework.

---

## 4. Robust regression: when outliers wreck the fit

### 4.1 The problem

Suppose the data mostly follow a nice trend, but one point is a gross **outlier** (a sensor glitch, a typo, a fat-fingered data entry). With the squared error, that single point can **tilt the whole line** toward itself. (See **Figure 5**, `figures/fig5_outlier_fit.png`: the squared-error line is dragged up toward the outlier, while the robust line stays with the trend.)

Why does this happen? Because the squared error $r_i^2$ grows *quadratically* with the residual $r_i = w^\top x_i - y_i$. A residual of $10$ contributes $100$ to the loss; the optimizer will do almost anything to shrink it, even at the cost of fitting the other 99 points slightly worse.

> **Is that bad?** It depends. If an outlier means *"a plane crashed"* you may *want* to pay attention to it. If it means *"someone mistyped a value"* you want to ignore it. Robust regression is for the second case.

### 4.2 The fix: absolute error ($L_1$ of the residuals)

Measure the training error with the **absolute** value instead of the square:

$$
\mathcal{L}(w) \;=\; \sum_{i=1}^{n} \big|\,w^\top x_i - y_i\,\big| \;=\; \lVert Xw - y\rVert_1 .
$$

> **Heads up — two different $L_1$'s!** In §2 the $L_1$ was on the **weights** $w$ (a regularizer that creates sparsity). Here the $L_1$ is on the **residuals** $Xw-y$ (a loss that creates robustness). Same norm, completely different job. Don't mix them up.

Now a residual of $10$ contributes only $10$, not $100$. Big and small errors are treated on a more equal footing, so one outlier can no longer dominate.

The deep reason is the **influence** of a point — how hard it pulls on the fit, which is the *derivative* of the loss w.r.t. the residual. (See **Figure 4**, `figures/fig4_robust_losses.png`, right panel.)

- Squared error: influence $= r_i$, which **grows without bound**. Far-away points pull hardest.
- Absolute error: influence $= \operatorname{sign}(r_i) = \pm 1$, **bounded**. A point 100 units away pulls exactly as hard as a point 2 units away.

Bounded influence = robustness. That's the whole secret.

### 4.3 Why absolute error is harder to minimize

Two annoyances compared with the squared error:

1. **No normal equations.** For the squared loss we could set the gradient to zero and *solve a linear system*. The $L_1$ loss has no such closed form.
2. **Non-differentiable at zero.** The absolute value $|r|$ has a **kink** at $r=0$: its slope is $-1$ for $r<0$ and $+1$ for $r>0$, with no well-defined derivative at $0$.

Non-smooth objectives are generally harder for gradient methods, because near a minimizer the gradient does **not** shrink toward zero (it jumps between $\pm 1$), so plain gradient descent tends to bounce around instead of settling. The standard trick is to replace the sharp kink with a **smooth approximation**.

### 4.4 The Huber loss: robust *and* smooth

The **Huber loss** is the best-of-both-worlds compromise. It behaves like the **squared** error for small residuals (smooth, nice gradients near the optimum) and like the **absolute** error for large residuals (bounded influence, robust):

$$
h_\epsilon(r_i) \;=\;
\begin{cases}
\dfrac{1}{2}\,r_i^2, & \text{if } |r_i| \le \epsilon \quad(\text{squared, near zero})\\[2mm]
\epsilon\Big(|r_i| - \dfrac{1}{2}\epsilon\Big), & \text{if } |r_i| > \epsilon \quad(\text{absolute, far away})
\end{cases}
\qquad
\mathcal{L}_{\text{huber}}(w) = \sum_{i=1}^{n} h_\epsilon\!\big(w^\top x_i - y_i\big).
$$

The threshold $\epsilon$ says where "small" ends and "large" begins. (See **Figure 4**, left panel: the purple Huber curve hugs the parabola near zero and the straight $|r|$ lines beyond $\pm\epsilon$.)

**Why the funny constant $-\tfrac12\epsilon$ in the second piece?** It is there to glue the two pieces together **continuously and smoothly** at $r=\pm\epsilon$. Let's verify at $r=\epsilon$:

- *Values match:* the quadratic piece gives $\tfrac12\epsilon^2$; the linear piece gives $\epsilon(\epsilon - \tfrac12\epsilon) = \epsilon\cdot\tfrac12\epsilon = \tfrac12\epsilon^2$. ✓ (same value, no jump)
- *Slopes match:* the quadratic piece has derivative $r$, which is $\epsilon$ at $r=\epsilon$; the linear piece has derivative $\epsilon$ (its slope). ✓ ($h'_\epsilon(\epsilon)=\epsilon$ and by symmetry $h'_\epsilon(-\epsilon)=-\epsilon$)

So $h_\epsilon$ is **differentiable everywhere** — including at the old kink. Two consequences:

- $\mathcal{L}_{\text{huber}}(w)$ is **convex** (one global minimum, no local traps), but $\nabla\mathcal{L}_{\text{huber}}(w)=0$ is **not a linear system**, so we still solve it iteratively.
- Optimizing it with **gradient descent is stable**, because the gradient *does* shrink toward zero near the minimizer (unlike the raw absolute error).

---

## 5. Two more variants: even more robust, and deliberately brittle

The loss is just a design choice, and you can push it in either direction. (See **Figure 6**, `figures/fig6_robustness_zoo.png`.)

### 5.1 Very robust regression: the square-root error (non-convex)

If bounded influence isn't enough, use an error that grows *even slower* than $|r|$, e.g. $\sqrt{|r_i|}$:

$$
\text{squared } r_i^2 \;\;\longrightarrow\;\; \text{absolute } |r_i| \;\;\longrightarrow\;\; \text{square-root } \sqrt{|r_i|}
$$

Reading left to right, each cares **less** about large residuals, so it is **more robust** to outliers. The square-root error barely reacts to a far-away point at all.

**The catch: non-convexity.** The square-root error is **not convex**. A non-convex objective can have **multiple local minima**, and gradient descent might land in a bad one — some of which might even *fit the outliers* (the very thing we were trying to avoid). The right panel of Figure 6 contrasts a convex bowl (one global minimum) with a wavy non-convex landscape (many local minima).

> **The rule to remember:** the **absolute error is the most robust *convex* loss.** Go beyond it (square-root, Tukey biweight, etc.) and you trade guaranteed optimization for extra robustness.

### 5.2 Brittle regression: the $L_\infty$ error

The opposite extreme. What if you *only* care about your **worst** prediction — e.g. a single catastrophic error is unacceptable? Then minimize the largest residual, the $L_\infty$ norm:

$$
\mathcal{L}(w) \;=\; \lVert Xw - y\rVert_\infty \;=\; \max_i \, |r_i| .
$$

Notes on this one:

- It is **very sensitive to outliers** ("brittle") — the single largest error controls everything — but that is exactly the point when the worst case is what matters.
- Unlike every other loss here, it is **not a sum or average** over the points; it's a max.
- It is **convex but not smooth** (a max of linear things has kinks). A smooth approximation is the log-sum-exp trick,
$$
\mathcal{L}(w) \;\approx\; \log \sum_{i=1}^{n} \exp\!\big(|w^\top x_i - y_i|\big),
$$
which you can then optimize with gradient descent.

### 5.3 The robustness spectrum at a glance

$$
\underbrace{L_\infty}_{\text{brittle}} \;\;\rightarrow\;\; \underbrace{L_2 \ (\text{squared})}_{\text{not robust}} \;\;\rightarrow\;\; \underbrace{L_1 \ (\text{absolute})}_{\substack{\text{most robust}\\ \text{convex loss}}} \;\;\rightarrow\;\; \underbrace{\sqrt{|r|}}_{\substack{\text{very robust}\\ \text{but non-convex}}}
$$

---

## 6. Summary — one page to rule them all

**Regularizers (penalties on the weights $w$, controlling model complexity):**

| Penalty | Formula | Effect | Optimization |
|---|---|---|---|
| $L_0$ | $\#\{j: w_j\neq0\}$ | true feature selection | non-convex, intractable |
| $L_1$ (LASSO) | $\lambda\lVert w\rVert_1$ | shrink **and** select (sparse) | convex, iterative |
| $L_2$ (ridge) | $\tfrac{\lambda}{2}\lVert w\rVert^2$ | shrink smoothly, keep all | convex, **closed form** |

**Losses (how we measure the training error on the residuals $r=Xw-y$):**

| Loss | Formula | Character |
|---|---|---|
| $L_\infty$ | $\max_i|r_i|$ | brittle: only the worst point matters |
| squared | $\tfrac12\lVert r\rVert^2$ | smooth, closed form, **not** robust |
| Huber | $\sum_i h_\epsilon(r_i)$ | smooth **and** robust (best default for noisy data) |
| absolute | $\lVert r\rVert_1$ | most robust **convex** loss; non-smooth |
| square-root | $\sum_i\sqrt{|r_i|}$ | very robust, but non-convex |

**Three mantras:**

1. *Regularizer near zero decides sparsity.* Constant slope down to zero ($L_1$) ⇒ exact zeros; vanishing slope ($L_2$) ⇒ small-but-non-zero.
2. *Bounded influence decides robustness.* A loss whose derivative stays bounded (absolute, Huber) ignores outliers; a loss with unbounded influence (squared) chases them.
3. *Smoothness and convexity decide how easy it is to optimize.* Smooth + convex (squared, ridge, Huber) is easy; non-smooth (absolute, $L_\infty$) needs care; non-convex (square-root) has no global guarantee.

**Where this is heading (Tutorial 2).** Every loss and every regularizer here has a hidden **probabilistic meaning**: the squared loss secretly assumes Gaussian noise, the absolute loss assumes Laplace noise, and the $L_2$/$L_1$ regularizers are secretly Gaussian/Laplace **priors** on the weights. That bridge — losses ⇄ likelihoods, regularizers ⇄ priors — is exactly the **MLE / MAP** story of the second tutorial.

---

### Figure index

| Figure | File | Shows |
|---|---|---|
| 1 | `figures/fig1_norm_penalties.png` | how $L_0$, $L_1$, $L_2$ charge one weight (near-zero behaviour) |
| 2 | `figures/fig2_sparsity_geometry.png` | diamond vs. circle → why $L_1$ hits a corner (sparsity) |
| 3 | `figures/fig3_regularization_path.png` | $L_2$ smooth shrink vs. $L_1$ exact zeros as $\lambda$ grows |
| 4 | `figures/fig4_robust_losses.png` | squared/absolute/Huber losses and their influence |
| 5 | `figures/fig5_outlier_fit.png` | one outlier: least-squares line dragged vs. robust line |
| 6 | `figures/fig6_robustness_zoo.png` | squared→absolute→square-root, and convex vs. non-convex |
