# Kernels and the Representer Theorem
### A tutorial that actually explains *why*

This is not a summary of the slides. It's an attempt to make you *understand* kernel methods — to the point where, if you forgot every formula, you could reinvent them. We'll follow one storyline and let each idea grow out of a problem the previous idea couldn't solve.

![The storyline](figures/story_arc.png)

We start with a wall that ordinary linear models hit, discover a trick that gets around it for free, ask what that trick *really* means (it secretly names an entire space of functions), and finish with the theorem that makes learning in that space possible at all. By the end, the famous formula

$$
f^{*}(x) = \sum_{i=1}^{n}\alpha_i\, k(x,x_i)
$$

should feel not clever but *inevitable*.

> **How to read this.** Each part opens with a question. Try to answer it yourself before reading on — the payoff of an idea is much bigger when you've felt the problem it solves. Boxes labelled **Intuition**, **Aha**, **Check yourself**, and **Common confusion** are where the real teaching happens; don't skip them.

**What you need to know already:** vectors and matrices, the dot product, gradients, and least-squares line fitting. Nothing about functional analysis — we build that here.

**Notation** (glance now, return later):

| Symbol | Read it as |
|---|---|
| $x, z$ | input points (e.g. in $\mathbb{R}^d$) |
| $\phi(x)$ | the "features" of $x$ (possibly infinite-dimensional) |
| $k(x,z)$ | kernel = similarity of $x$ and $z$ = $\langle\phi(x),\phi(z)\rangle$ |
| $\langle f,g\rangle,\ \|f\|$ | inner product and norm (of vectors *or* functions) |
| $\mathcal{F}$ (or $\mathcal H$) | the feature space / RKHS |
| $K$ | kernel matrix, $K_{ij}=k(x_i,x_j)$ |
| $\alpha$ | the $n$ coefficients we actually solve for |

---

# Part I · The trick that shouldn't work

## 1. The wall: a straight line can't bend

Linear models are the best-understood objects in all of machine learning. A linear model

$$
f(x) = w^{\top}x
$$

is fast to train, has a single global optimum, and you can reason about exactly what it does. There's just one problem: it is **rigid**. Its decision boundary is a flat line (or plane). Give it data shaped like a target — one class in the middle, the other in a ring around it — and it fails completely. No straight line puts the blob on one side and the ring on the other.

![The wall, and a way over it](figures/lift_circle.png)

The left panel is the wall. The question that launches everything is:

> **The driving question.** Can we keep the wonderful machinery of linear models — the convexity, the closed-form solutions — but somehow let them bend?

## 2. The obvious idea, and the bill it runs up

Here's the natural fix. If the data isn't linearly separable *as given*, invent new coordinates in which it is. Replace each point $x$ by a richer feature vector $\phi(x)$, and run the ordinary linear model on $\phi(x)$ instead of $x$:

$$
f(x) = w^{\top}\phi(x).
$$

This is still **linear in the parameters** $w$ (so all the nice machinery survives), but it can be wildly **nonlinear in $x$**.

Take the target-shaped data and the feature map

$$
\phi(x_1,x_2) = \big(x_1^2,\ \sqrt{2}\,x_1x_2,\ x_2^2\big).
$$

A flat plane in this 3-D feature space, $w^{\top}\phi(x)=c$, corresponds back in the original space to $w_1x_1^2 + w_2\sqrt2\,x_1x_2 + w_3x_2^2 = c$ — the equation of a **conic** (circle, ellipse…). The right panel above shows exactly this: after lifting, a plane slices the blob cleanly from the ring. We taught a linear model to draw a circle.

So features are the answer. Just use *lots* of them. And that's where the bill arrives.

**How many features does "lots" cost?** For degree-$p$ polynomial features in $d$ input dimensions, the number of distinct monomials is $\binom{d+p}{p}$. That grows explosively.

![The cost of explicit features](figures/feature_explosion.png)

With $d=100$ inputs (small by modern standards) and degree $p=4$, you already need on the order of *four million* features. Push the degree up, or use a feature map that is genuinely infinite-dimensional, and computing $\phi(x)$ — let alone storing $w$ — becomes flat-out impossible.

> **Where we're stuck.** Rich features give linear models the power to bend, but rich features are unaffordable. We seem forced to choose between *expressive* and *computable*. Part I's punchline is that we don't have to.

## 3. The escape hatch: you never needed the features

Look very closely at what a feature-based linear method actually *does* with $\phi$. Here is the observation the whole field is built on:

**In an enormous class of algorithms — ridge regression, SVMs, PCA, and more — the feature vectors only ever appear inside inner products $\langle\phi(x),\phi(z)\rangle$.** Never alone. (We'll make this airtight in Part III with the representer theorem; for now, notice that predictions, distances, and angles are all built from dot products.)

If we only ever need the *dot product* of two feature vectors, maybe we can compute that dot product directly — without ever forming the vectors. Let's test it on our degree-2 map. Take two points $x$ and $z$ and grind out $\langle\phi(x),\phi(z)\rangle$:

$$
\begin{aligned}
\langle\phi(x),\phi(z)\rangle
&= (x_1^2)(z_1^2) + (\sqrt2\,x_1x_2)(\sqrt2\,z_1z_2) + (x_2^2)(z_2^2)\\[2pt]
&= x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2\\[2pt]
&= (x_1z_1 + x_2z_2)^2\\[2pt]
&= (x^{\top}z)^2.
\end{aligned}
$$

Read that last line slowly. The dot product of two **three-dimensional** feature vectors is exactly $(x^{\top}z)^2$ — a formula that only touches the **original two-dimensional** inputs and costs almost nothing. We computed a similarity in feature space *without visiting feature space*.

Give this shortcut a name.

> **Definition (kernel).** A **kernel** $k(x,z)$ is a function that returns $\langle\phi(x),\phi(z)\rangle$ for some feature map $\phi$. It is "the dot product in the lifted space, computed cheaply in the original space."

Now scale the trick up:

- $k(x,z) = (x^{\top}z)^p$ is the dot product of *all* degree-$p$ monomial features — $\binom{d+p}{p}$ of them — yet it costs one dot product and one power, i.e. $O(d)$ work.
- $k(x,z) = \exp\!\big(-\|x-z\|^2/2\sigma^2\big)$ — the **RBF (Gaussian) kernel** — corresponds to an **infinite-dimensional** feature map. (Taylor-expand the exponential and you get infinitely many polynomial features.) You could never write those features down. The kernel is one line of code.

![A kernel is a similarity score](figures/similarity_bumps.png)

The left panel shows the RBF kernel for what it is: a **similarity** that is $1$ when $x=z$ and fades with distance, at a rate set by the bandwidth $\sigma$. That's the mental image to keep — **a kernel measures how alike two points are.**

> **Aha.** You can measure lengths, angles, and distances in a million- or even infinite-dimensional space *that you never construct*, using a cheap formula on the raw inputs. This is the **kernel trick**, and it is why kernel methods are practical at all.

## 4. Which functions are allowed to be kernels?

We can't just declare any two-argument function a kernel — it has to genuinely be a dot product of *some* features. Which functions qualify? The answer is clean:

> **A function $k$ is a valid kernel $\iff$ it is symmetric and positive semi-definite (PSD).**
> - Symmetric: $k(x,z)=k(z,x)$.
> - PSD: for **any** finite set of points $x_1,\dots,x_m$, the **kernel matrix** $K$ with $K_{ij}=k(x_i,x_j)$ satisfies $c^{\top}Kc \ge 0$ for every vector $c$.

Why is PSD *exactly* the right condition? Suppose $k(x,z)=\langle\phi(x),\phi(z)\rangle$. Then

$$
c^{\top}Kc = \sum_{i,j}c_ic_j\,\langle\phi(x_i),\phi(x_j)\rangle
= \Big\langle \sum_i c_i\phi(x_i),\ \sum_j c_j\phi(x_j)\Big\rangle
= \Big\|\sum_i c_i\phi(x_i)\Big\|^2 \ge 0.
$$

A squared length can't be negative, so *being a dot product forces PSD*. The deep theorems of Part II (Moore–Aronszajn) give the converse: every symmetric PSD function really is a dot product of some features. So "symmetric PSD" is the precise membership card for the club of kernels.

Your working menu:

| Kernel | Formula | Secretly, the features are… |
|---|---|---|
| Linear | $k(x,z)=x^{\top}z$ | $x$ itself |
| Polynomial | $k(x,z)=(x^{\top}z+c)^{p}$ | all monomials up to degree $p$ |
| RBF / Gaussian | $k(x,z)=\exp(-\|x-z\|^2/2\sigma^2)$ | infinitely many |

> **Check yourself.** Is $k(x,z) = x^{\top}z - 1$ a valid kernel? *(Take a single point $x$ with $\|x\|^2<1$; then the $1\times1$ kernel matrix is $\|x\|^2-1<0$, so it's not PSD — not a kernel.)* The PSD test is not a formality; it really rejects things.

**Where we are.** We can now run linear methods inside gigantic feature spaces for pennies. But two honest questions remain, and they're the subject of the rest of the tutorial:

1. **What space of functions does a kernel actually give us?** (Part II)
2. **Out of that whole space, which single function does learning choose?** (Part III)

---

# Part II · The space hiding inside a kernel

## 5. Functions are vectors (get comfortable with this)

To ask "what space of functions does a kernel give," we first need to be at ease treating a **function as a vector**. This sounds abstract; it's not.

A vector in $\mathbb{R}^d$ is a list of $d$ numbers, and you can add two of them and scale them. A function $f$ is like a list of numbers too — one number $f(x)$ for **every** input $x$ — and you can add and scale functions in the obvious pointwise way:

$$
[f+g](x) = f(x)+g(x), \qquad [a\cdot f](x) = a\,f(x).
$$

So functions form a vector space; a function is just a vector with (usually infinitely) many coordinates. And wherever there are vectors, we can put **geometry** on them with an inner product — the continuous cousin of the dot product:

$$
\langle f,g\rangle = \int f(x)\,g(x)\,dx, \qquad \|f\| = \sqrt{\langle f,f\rangle}.
$$

(Compare: the dot product multiplies matching coordinates and *sums*; this multiplies matching values $f(x)g(x)$ and *integrates*. Same idea, infinitely many coordinates.) With an inner product we get lengths, distances, and angles between functions. Everything you know about vector geometry now applies to functions.

## 6. The one property we actually need — and why it's not free

Here's the thing we're secretly going to rely on. After learning a function $f$, we **use** it by evaluating $f(x)$ at new points. For that to be trustworthy, "evaluate at $x$" must be a *stable* operation:

> if two functions are close (small $\|f-g\|$), their values $f(x)$ and $g(x)$ should be close too.

Call this **continuity of evaluation**. It feels like it should always hold. It does not.

![Why plain $L^2$ is not good enough](figures/eval_not_continuous.png)

Look at these shrinking triangular spikes. Each has height $1$ at the origin, but they get narrower. Their $L^2$ size, $\|f\|_2^2=\tfrac{2w}{3}$, goes to zero as the width $w\to 0$ — the functions become *negligible* in the $L^2$ sense. Yet every single one has $f(0)=1$. So in $L^2$ you can have $\|f\|\to 0$ while the value at a point stubbornly stays at $1$: **evaluation is not continuous.** (The deeper reason: in $L^2$, functions are only pinned down "almost everywhere," so the value $f(0)$ at one exact point isn't even well-defined. To pick out a single point's value you'd need a Dirac spike, which isn't a legit $L^2$ function.)

This is the very gap that the **Sobolev** inner product plugs. Recall the Sobolev space adds a derivative term, $\langle f,g\rangle_{H^1}=\int fg + \int f'g'$. Controlling $\int (f')^2$ too forbids those tall thin spikes (a spike has enormous slope), which is exactly what makes evaluation continuous again. Different kernels are different ways of controlling functions enough to make pointwise values safe.

We give the well-behaved spaces a name:

> **Definition (RKHS).** A **Reproducing Kernel Hilbert Space** is a Hilbert space of functions in which, for every point $x$, the evaluation map $\operatorname{eval}_x : f\mapsto f(x)$ is **continuous** — equivalently, there is a constant $C_x$ with $|f(x)| \le C_x\,\|f\|$ for all $f$.

That inequality is the whole content: it says a function's value at a point can't be large unless the function's norm is large. Evaluation can't "blow up." **That is the entire reason RKHS is the right home for learning — it's the class of function spaces where making a prediction is a safe operation.**

## 7. Riesz hands you the kernel (the magic step)

Now watch the kernel appear out of pure structure. It rests on one classical fact:

> **Riesz representation.** In a Hilbert space, every *continuous linear measurement* of vectors is secretly a dot product with a single fixed vector.

The finite-dimensional version is familiar: any linear gauge on $\mathbb{R}^d$, $v\mapsto a^{\top}v$, is "dot with $a$." Riesz says the same is true in infinite dimensions, for continuous linear functionals.

Apply it to evaluation. The map $\operatorname{eval}_x:f\mapsto f(x)$ is linear, and in an RKHS it is continuous — so Riesz applies. There must be a fixed function, call it $k_x$, such that

$$
\boxed{\,f(x) = \langle f,\ k_x\rangle \quad\text{for every } f.\,}
$$

This is the **reproducing property**: dotting any function with the special function $k_x$ "reproduces" its value at $x$. Now do the one clever thing — the function $k_x$ is itself a member of the space, so evaluate *it* at another point $z$, using the very same property:

$$
k_x(z) = \langle k_x,\ k_z\rangle.
$$

Define the kernel and the feature map by

$$
k(x,z) := k_x(z) = \langle k_x, k_z\rangle,
\qquad
\phi(x) := k_x .
$$

Then $k(x,z)=\langle\phi(x),\phi(z)\rangle$ — precisely the "dot product of features" from Part I. The circle closes: **the cheap similarity function we invented is the reproducing kernel of an honest Hilbert space of functions, and the mysterious feature map is $\phi(x)=k_x=k(\cdot,x)$ — the "similarity-to-$x$ bump."**

![Functions are weighted sums of similarity bumps](figures/similarity_bumps.png)

The right panel is worth internalizing: because $\phi(x)=k(\cdot,x)$ is a bump centered at $x$, the natural functions in an RKHS are **weighted sums of bumps**, $f=\sum_i\alpha_i\,k(\cdot,x_i)$. Keep that image; the representer theorem is about to tell us the *best* learned function always has exactly this shape.

> **Common confusion.** "What's the feature vector $\phi(x)$ for the RBF kernel?" It isn't a list of numbers — it's a whole *function*, the bump $k(\cdot,x)$. In an infinite-dimensional feature space, "vectors" are functions. You never build them; the kernel $k$ is the only thing you ever compute.

## 8. Moore–Aronszajn: a perfect two-way dictionary

We just walked one direction: an RKHS hands you a symmetric PSD reproducing kernel. The classical theorem gives the exact converse.

> **Moore–Aronszajn theorem (Aronszajn, 1950).** Every symmetric positive-semi-definite kernel $k$ is the reproducing kernel of **one and only one** RKHS.

Put the two directions together and you get a clean bijection:

$$
\text{symmetric PSD kernel } k \quad\Longleftrightarrow\quad \text{a unique RKHS } \mathcal{F}.
$$

This is the rigorous licence for everything in Part I. **Choosing a kernel is the same act as choosing a space of functions** — you get the whole (possibly infinite-dimensional) space, and all of its inner products, for free, just by writing down $k$. You never build the space; you compute with $k$ and Moore–Aronszajn guarantees the space is there, doing the geometry for you.

## 9. The hidden gift: the norm is a complexity meter

An RKHS gives you more than a bag of functions. It gives you a **norm** $\|f\|$, and that norm quietly measures how *complicated* a function is.

![The RKHS norm charges for wiggles](figures/norm_complexity.png)

The three sine waves above all have identical $L^2$ size ($\|f\|_2^2=\pi$). But look what a smoothness-aware norm does: the "roughness" $\int (f')^2$ equals $k^2\pi$, so it grows with the frequency $k$. A norm like this barely notices *amplitude* but heavily penalizes *wiggling*. For the RBF kernel the same is true: a function with **small RKHS norm is smooth and slowly varying**; a large norm means sharp, high-frequency structure.

Why does this matter so much? Because it turns the vague wish "prefer a simpler function" into a precise, minimizable number: **prefer a function with small $\|f\|$.** That is the hinge the entire next part swings on.

> **The one sentence to remember about RKHS.** A kernel gives you a space of functions *and* a built-in ruler for how complicated each one is. Similarity, feature space, and complexity meter — all three are the same object, $k$.

---

# Part III · Learning in an infinite space

## 10. Stating the learning problem honestly

We have training data $(x_1,y_1),\dots,(x_n,y_n)$ and we want a function $f$ in our RKHS that does two things at once: **fits the data** and **stays simple**. Written as a single objective, this is the **regularized empirical risk**:

$$
f^{*} = \arg\min_{f\in\mathcal{F}}\ \underbrace{\frac1n\sum_{i=1}^{n}L\big(f(x_i),\,y_i\big)}_{\text{fit the data}}
\ +\ \underbrace{g(\|f\|)}_{\text{stay simple}},
$$

where $L$ is any loss (squared error, hinge, logistic, …) and $g$ is a **strictly increasing** function of the norm (e.g. $g(\|f\|)=\lambda\|f\|^2$).

Why is the second term non-negotiable? Because the RKHS can be infinite-dimensional. Without a penalty on complexity you can fit the training points *perfectly* in infinitely many wild ways — the problem is ill-posed and you overfit catastrophically. The norm penalty, using the complexity meter from §9, breaks the tie by choosing the *simplest function that fits well enough*.

![Same model, three regularization strengths](figures/lambda_effect.png)

The knob $\lambda$ (inside $g$) controls the trade: too little and the model memorizes noise, too much and it forgets the signal, and in between it learns the true shape. (More on choosing $\lambda$ in §14.)

## 11. The impossibility — and the theorem that erases it

To minimize that objective we must, in principle, search over **every function in a possibly infinite-dimensional space**. That should be hopeless. It isn't, because of one of the most important results in machine learning.

> **Representer theorem** (Kimeldorf & Wahba, 1971; general form Schölkopf, Herbrich & Smola, 2001). For the regularized empirical risk above with any loss $L$ and any strictly increasing $g$, the minimizer is a **finite** combination of the training bumps:
> $$
> f^{*}(\cdot) = \sum_{i=1}^{n}\alpha_i\,k(\cdot, x_i),\qquad \alpha_i\in\mathbb{R}.
> $$

The infinite search **collapses to $n$ numbers** $\alpha_1,\dots,\alpha_n$ — one per data point. This single fact is what makes kernel learning computable, and it's why the "sum of bumps" picture from §7 kept showing up.

## 12. Why it's true — and the idea is genuinely simple

You can understand the whole proof through one picture and two questions.

Take any candidate function $f$ and split it into two pieces: the part $u$ that lives in the span of the training bumps $\{k(\cdot,x_1),\dots,k(\cdot,x_n)\}$, and whatever is **left over**, $v$, which is orthogonal to every one of those bumps:

$$
f = u + v,\qquad u\in S:=\operatorname{span}\{k(\cdot,x_i)\},\qquad v\perp S .
$$

![The one picture behind the proof](figures/orthogonal_decomposition.png)

Now interrogate the leftover $v$ with two questions.

**Question 1 — does $v$ change any prediction?** Evaluate $f$ at a training point $x_j$ with the reproducing property, then split:

$$
f(x_j)=\langle f, k(\cdot,x_j)\rangle=\langle u, k(\cdot,x_j)\rangle+\underbrace{\langle v, k(\cdot,x_j)\rangle}_{=\,0}.
$$

The second term is zero because $k(\cdot,x_j)$ is one of the bumps — it lives in $S$ — and $v$ is orthogonal to all of $S$. So $f(x_j)=u(x_j)$: **the leftover $v$ is completely invisible to the data.** The fitting term literally cannot tell $u+v$ apart from $u$.

**Question 2 — does $v$ cost anything?** Because $u$ and $v$ are orthogonal, Pythagoras gives

$$
\|f\|^2 = \|u\|^2 + \|v\|^2 \ \ge\ \|u\|^2 ,
$$

with equality only when $v=0$. Since $g$ is strictly increasing, any nonzero $v$ **strictly raises the penalty** $g(\|f\|)$.

**Put them together.** The leftover $v$ helps the fit not at all (Q1) and costs extra penalty (Q2). So the optimizer will always set $v=0$. What survives, $u$, lives in the span of the bumps — which is precisely $f^{*}(\cdot)=\sum_i\alpha_i\,k(\cdot,x_i)$. $\blacksquare$

> **The whole proof in one sentence.** The part of a function orthogonal to the data is invisible to the fit but visible to the penalty — so the best function throws it away, leaving a finite sum of kernels.

## 13. Cashing it out: kernel ridge regression, start to finish

Let's turn the theorem into a concrete algorithm and watch the promises from Parts I–II all pay off at once. Choose the **squared loss** and the penalty $g(\|f\|)=\lambda\|f\|^2$:

$$
f^{*}=\arg\min_{f\in\mathcal{F}}\ \frac1n\sum_{i=1}^n\big(f(x_i)-y_i\big)^2 + \lambda\|f\|^2 .
$$

**Step 1 — use the theorem.** We know $f^{*}=\sum_{i}\alpha_i\,k(\cdot,x_i)$, so we only need the vector $\alpha\in\mathbb{R}^n$.

**Step 2 — rewrite in $\alpha$ and $K$.** Let $K$ be the kernel matrix, $K_{ij}=k(x_i,x_j)$. Two facts:

- Predictions: $f^{*}(x_j)=\sum_i\alpha_i k(x_j,x_i)=(K\alpha)_j$, so the vector of fitted values is $K\alpha$.
- Norm: $\|f^{*}\|^2=\big\langle\sum_i\alpha_i\phi(x_i),\sum_j\alpha_j\phi(x_j)\big\rangle=\sum_{i,j}\alpha_i\alpha_j\,k(x_i,x_j)=\alpha^{\top}K\alpha$.

Substitute (and use $K^{\top}=K$):

$$
h(\alpha)=\frac1n\|K\alpha-y\|^2+\lambda\,\alpha^{\top}K\alpha
=\frac1n\big[\alpha^{\top}K^2\alpha-2\alpha^{\top}Ky+y^{\top}y\big]+\lambda\,\alpha^{\top}K\alpha .
$$

**Step 3 — set the gradient to zero.** Using $\nabla_\alpha(\alpha^{\top}M\alpha)=2M\alpha$ for symmetric $M$:

$$
\nabla_\alpha h = \frac1n(2K^2\alpha-2Ky)+2\lambda K\alpha \overset{!}{=}0
\ \Longrightarrow\ K^2\alpha+n\lambda K\alpha=Ky
\ \Longrightarrow\ K(K+n\lambda I)\alpha=Ky.
$$

Cancel a factor of $K$ (legitimate when $K$ is invertible) to land at the clean result:

$$
\boxed{\,\alpha^{*}=(K+n\lambda I)^{-1}\,y\,}
$$

**Step 4 — predict at a new point.** With the row of similarities $k(x,X)=\big(k(x,x_1),\dots,k(x,x_n)\big)$,

$$
\boxed{\,f^{*}(x)=k(x,X)\,(K+n\lambda I)^{-1}\,y\,}
$$

![Kernel ridge regression, computed](figures/kernel_ridge_fit.png)

The bold curve is $f^{*}$; the faint curves are the individual weighted bumps $\alpha_i^{*}\,k(\cdot,x_i)$ that sum to it — the representer theorem made visible.

Now notice the two things that make this remarkable:

- **Nothing but kernels appears.** No $\phi$, no feature-space dimension, no $w$. Every quantity — $K$, $k(x,X)$ — is just kernel evaluations. Part I's promise (work in a huge space via cheap dot products) is fully realized, and Part III *guarantees* this finite formula is exactly the optimum, not an approximation.
- **The regularizer fixes the algebra too.** $K+n\lambda I$ is invertible for any $\lambda>0$ (it shifts every eigenvalue of the PSD matrix $K$ up by $n\lambda>0$), so regularization simultaneously fights overfitting *and* makes the linear system well-posed.

> **Try it in your head (tiny example).** With two identical points $x_1=x_2$ and an RBF kernel, $K=\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ is singular — ordinary least squares chokes. But $K+n\lambda I$ has eigenvalues $2+n\lambda$ and $n\lambda$, both positive, so kernel ridge sails through. That's regularization earning its keep.

## 14. Reality checks (the things slides skip)

A tutorial that only sells the method is lying to you. Here's the honest operating manual.

- **Cost.** Solving for $\alpha^{*}$ is $O(n^3)$ and storing $K$ is $O(n^2)$. So kernels shine when features vastly outnumber data ($p\gg n$) and *struggle* when $n$ is huge. For big $n$, people use approximations (Nyström, random Fourier features) that trade a little accuracy for scalability.
- **The kernel is your inductive bias.** Picking a kernel is picking *what "similar" means* and therefore what functions you consider simple. RBF assumes smoothness; a polynomial kernel assumes polynomial structure. This choice matters more than almost anything else.
- **Two knobs, set by validation.** The RBF bandwidth $\sigma$ controls how far influence spreads; $\lambda$ controls the fit-vs-simplicity trade. Tune both on held-out data (cross-validation), not on the training set.
- **What the solution *is*, intuitively.** $f^{*}(x)=\sum_i\alpha_i k(x,x_i)$ says: *to predict at $x$, ask how similar $x$ is to each training point, and combine the training answers weighted by those similarities and by $\alpha$.* It's a smart, learned form of "nearest-neighbor by similarity."

---

# The one idea to carry away

Everything above is a single thought seen from different angles:

> **A kernel is three things at once** — a cheap similarity score, a hidden (possibly infinite-dimensional) feature space, and a ruler for how complex a function is. **Regularized learning in that space always collapses to $n$ numbers** (the representer theorem), **computed with the kernel alone.**

Kernel ridge regression, support-vector machines, Gaussian-process regression, kernel PCA — they are all this one pattern with a different loss or a different downstream use. Once you see the pattern, you've seen the field.

---

## Check your understanding

Try these before peeking at the answers.

1. **Why can a linear model with features $\phi$ draw a circle, even though it's "linear"?**
2. **What does the PSD condition on a kernel actually guarantee about the feature map?**
3. **In one sentence, why is $L^2$ not an RKHS but a Sobolev space is?**
4. **In the representer-theorem proof, which term "sees" the orthogonal component $v$, and which one is blind to it?**
5. **In $\alpha^{*}=(K+n\lambda I)^{-1}y$, what breaks if $\lambda=0$ and two data points coincide, and why does $\lambda>0$ fix it?**

<details>
<summary><strong>Answers</strong></summary>

1. It's linear *in the parameters* $w$ and in the *feature coordinates* $\phi(x)$, but $\phi$ is nonlinear in $x$; a flat boundary in feature space bends into a curved boundary in input space.
2. It guarantees $k$ really is a dot product of *some* features: $c^\top K c=\|\sum_i c_i\phi(x_i)\|^2\ge 0$, so a valid feature map exists (Moore–Aronszajn).
3. In $L^2$ a function can have vanishing norm yet value $1$ at a point (shrinking spikes), so evaluation isn't continuous; adding the derivative term in the Sobolev norm forbids those spikes and makes evaluation continuous — an RKHS.
4. The *penalty* $g(\|f\|)$ sees $v$ (it adds $\|v\|^2$ to the norm); the *data-fit* term is blind to it (predictions depend only on $u$). So the optimum sets $v=0$.
5. With coinciding points $K$ is singular, so $(K)^{-1}$ (i.e. $\lambda=0$) doesn't exist; $K+n\lambda I$ shifts all eigenvalues up by $n\lambda>0$, making it invertible.

</details>

---

## References and further reading

The theory here is standard; these are the primary sources and the best places to go deeper.

1. N. Aronszajn (1950). *Theory of Reproducing Kernels.* Transactions of the American Mathematical Society, **68**(3), 337–404. — The Moore–Aronszajn theorem; the origin of RKHS theory.
2. G. Kimeldorf & G. Wahba (1971). *Some results on Tchebycheffian spline functions.* Journal of Mathematical Analysis and Applications, **33**, 82–95. — The original representer theorem (squared-error case).
3. B. Schölkopf, R. Herbrich & A. Smola (2001). *A Generalized Representer Theorem.* COLT 2001, LNCS 2111, 416–426. — Extends it to any loss and any strictly increasing regularizer (the version used here).
4. J. Mercer (1909). *Functions of positive and negative type…* Phil. Trans. Royal Society A, **209**. — Mercer's theorem, the spectral view of PSD kernels.
5. M. Aizerman, E. Braverman & L. Rozonoer (1964). *Theoretical foundations of the potential function method…* — Early appearance of the kernel trick; later revived by B. Boser, I. Guyon & V. Vapnik (1992) for support-vector machines.
6. B. Schölkopf & A. Smola (2002). *Learning with Kernels.* MIT Press. — The standard, highly readable textbook on kernel methods.
7. I. Steinwart & A. Christmann (2008). *Support Vector Machines.* Springer. — Rigorous modern treatment of RKHS and learning theory.
8. Wikipedia, [*Reproducing kernel Hilbert space*](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) and [*Representer theorem*](https://en.wikipedia.org/wiki/Representer_theorem) (accessed July 2026). — Accessible summaries with references, consulted to confirm statements and citations above.
9. A. Lehrmann, *Machine Learning — Kernel Theory* and *— Regularization* (THWS course slides). — The lectures this tutorial expands and re-explains.

*All figures in this document were generated programmatically with Python/matplotlib (script included in the bundle) so that every curve, count, and contour is numerically honest rather than sketched.*
