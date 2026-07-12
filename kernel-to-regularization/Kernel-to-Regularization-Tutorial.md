# From Kernels to Regularization
### A beginner-friendly tutorial, built from the ground up

*Based on the lecture decks **"Machine Learning — Kernel Theory"** and **"Machine Learning — Regularization"** (A. Lehrmann, THWS). This tutorial keeps the exact logical flow of the slides but fills in every gap — especially the math — and adds intuition and pictures so you can actually learn it, not just recognize it.*

---

## How to read this

The two lectures tell **one continuous story**. It goes like this:

![Roadmap](figures/roadmap.png)
*The whole journey on one line. We start from an abstract object (a kernel), discover it secretly lives in a special space of functions (an RKHS), use that structure to prove a theorem that makes learning tractable (the representer theorem), and finally turn it into a usable algorithm (kernel ridge regression). Regularization is the glue that makes all of it generalize.*

Each section builds on the previous one, so read in order. Boxes like the one below are **intuition first, formalism second** — read the box, then the math will feel inevitable.

> **In plain words.** A *kernel* is a function that measures how similar two data points are. The astonishing fact this whole tutorial builds toward: choosing a similarity function is *secretly the same thing* as choosing an infinite-dimensional space of features — and you never have to touch that space directly.

**Prerequisites:** vectors, matrices, dot products, gradients, and the idea of fitting a line by minimizing squared error. Everything else is built here.

**Notation cheat-sheet** (skim now, refer back later):

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$ | input space (where data points $x$ live, often $\mathbb{R}^d$) |
| $\mathcal{F}$ | feature space (a Hilbert space of functions) |
| $\phi:\mathcal{X}\to\mathcal{F}$ | feature map (lifts a point into feature space) |
| $k(x,x')$ | kernel: a number measuring similarity of $x$ and $x'$ |
| $\langle\cdot,\cdot\rangle$ | inner product (generalized dot product) |
| $\|v\| = \sqrt{\langle v,v\rangle}$ | norm (length) |
| $w$ | weight vector of a linear model |
| $\alpha$ | dual coefficients (weights on data points) |
| $K = K(X,X)$ | kernel matrix, $K_{ij}=k(x_i,x_j)$ |
| $n,\ d$ | number of training points, number of features |
| $\lambda$ | regularization strength |

---

# Part 0 — Recap: where we left off

Last time we did **linear classification**: take the output of a linear regression model $w^{\top}x$ and map it to a label.

$$
\hat{y}_i = \operatorname{sign}(w^{\top}x_i)\in\{-1,+1\}\quad(\textbf{deterministic}),
\qquad
\hat{y}_i = p(y_i=+1\mid x_i)=\sigma(w^{\top}x_i)\in[0,1]\quad(\textbf{probabilistic}),
$$

where $\sigma(z)=\dfrac{1}{1+e^{-z}}$ is the logistic (sigmoid) function.

**Which loss do we minimize?** Squared error is great for regression but *wrong* for classification: it punishes predictions that are confidently correct. Instead we use one of these two, written in terms of the **margin** $m = y_i\,w^{\top}x_i$ (positive = correct and confident):

$$
\mathcal{L}_{\text{hinge}}(w)=\sum_{i=1}^{n}\max\{0,\,1-y_i w^{\top}x_i\},
\qquad
\mathcal{L}_{\text{log}}(w)=\sum_{i=1}^{n}\log\!\big(1+\exp(-y_i w^{\top}x_i)\big).
$$

![Loss functions](figures/losses.png)
*The ideal classification loss is the 0–1 loss (grey): pay 1 for a mistake, 0 otherwise. But it is flat and discontinuous, so gradient methods can't optimize it. Hinge (blue) and logistic (orange) are smooth, convex **surrogates** that go to zero as the margin grows. Squared loss (red dashed) is a bad choice — it blows up even when a point is correctly classified with a large margin.*

**The punchline that sets up everything to come:** *kernels* (e.g. RBF, polynomial) let us do linear classification **efficiently in very high-dimensional feature spaces**. The trick is to optimize a vector $\alpha$ of weights-on-data-points instead of a weight vector $w$, using the relationship

$$
w = Z^{\top}\alpha,\qquad Z = \big(\phi(x_1),\dots,\phi(x_n)\big)^{\top}.
$$

Why is that a good idea? Because $w$ may live in a space with millions (or infinitely many) dimensions, but $\alpha$ only has $n$ entries — one per training point. **This tutorial explains why we are allowed to do that, and what the mysterious $\phi$ and $k$ really are.**

> **Why bother lifting to a high-dimensional space at all?** Because data that a straight line can't separate often becomes separable after a nonlinear lift.
>
> ![Kernel trick](figures/kernel_trick.png)
> *Left: two classes arranged as an inner blob and an outer ring. No straight line separates them. Right: apply the feature map $\phi(x)=(x_1^2,\ \sqrt{2}\,x_1x_2,\ x_2^2)$. In this lifted space the two classes fall on different "heights," and a flat plane separates them perfectly. A linear method in feature space = a nonlinear method in the original space.*

---

# Part 1 — The mathematical playground: linear algebra recap

Before we can talk about kernels we need the right stage. That stage is built from four nested ideas: **vector space → inner-product space → normed space → metric space**. Each one adds a new ability.

## 1.1 Vector spaces

> **Intuition.** A vector space is any collection of objects you can **add together** and **scale** by numbers, where these operations behave the way you'd expect. "Objects" doesn't have to mean arrows — it can mean functions, as we'll see next.

A **real vector space** ($\mathbb{R}$-vector space) is a set $V$ with two operations,

$$
+ : V\times V\to V \qquad\text{and}\qquad \cdot : \mathbb{R}\times V \to V,
$$

such that:

- **Addition** is associative and commutative, has an identity element $0$ (with $v+0=v$), and every $v$ has an inverse $-v$ (with $v+(-v)=0$).
- **Scalar multiplication** satisfies, for all $u,v\in V$ and $a,b\in\mathbb{R}$:
  - identity: $1\cdot v = v$,
  - compatibility with multiplication: $(ab)v = a(bv)$,
  - distributivity: $a(u+v)=au+av$ and $(a+b)v = av + bv$.

**The familiar example — $\mathbb{R}^d$.** The set $V=\{(v_1,\dots,v_d)\mid v_i\in\mathbb{R}\}$ with componentwise operations

$$
(u_1,\dots,u_d)+(v_1,\dots,v_d)=(u_1+v_1,\dots,u_d+v_d),
\qquad a\cdot(v_1,\dots,v_d)=(av_1,\dots,av_d)
$$

is the vector space $\mathbb{R}^d$. This is the "arrows" you already know.

## 1.2 Functions are vectors, too

Here is the conceptual leap the whole subject rests on.

> **Key idea.** A **function is a vector.** If you can add two functions and scale a function — and you can — then a set of functions is a vector space. Instead of a list of $d$ numbers, a function is like a list of *infinitely many* numbers: its value at every input point.

Let $\mathcal{X}$ be any set. The set of all real-valued functions on it,

$$
V=\{f:\mathcal{X}\to\mathbb{R}\},
$$

with the operations defined **pointwise**,

$$
[f+g](x)=f(x)+g(x),
\qquad
[a\cdot f](x)=a\cdot f(x),
$$

is an $\mathbb{R}$-vector space. (Check the axioms if you like — they follow immediately from the same axioms for real numbers.)

A more refined example we'll need is the space of **square-integrable functions**. Let $\mathcal{X}=[a,b]$ be a real interval and define

$$
V=\Big\{f:[a,b]\to\mathbb{R}\ \Big|\ \int_a^b f(x)^2\,dx < \infty\Big\}.
$$

With the same pointwise operations, this forms the vector space $L^2([a,b])$. The condition $\int f^2<\infty$ just says the function doesn't have "too much energy" — it keeps the integrals we're about to define finite.

## 1.3 Inner-product spaces

Adding and scaling isn't enough; we also want **geometry** — angles, lengths, "how aligned are these two vectors?" That is what an inner product provides.

> **Intuition.** An inner product is a generalized dot product: feed it two vectors, get back one number measuring how much they point the same way. Big positive = aligned; zero = orthogonal (perpendicular).

An **inner-product space** is a vector space $V$ with an operation $\langle\cdot,\cdot\rangle:V\times V\to\mathbb{R}$ satisfying:

- **symmetry:** $\langle u,v\rangle=\langle v,u\rangle$,
- **linearity:** $\langle au+bv,\,w\rangle = a\langle u,w\rangle + b\langle v,w\rangle$,
- **positive-definiteness:** $\langle v,v\rangle > 0$ for every $v\neq 0$.

From an inner product you get, for free, two more layers of structure:

$$
\underbrace{\|v\| = \sqrt{\langle v,v\rangle}}_{\textbf{norm (length)}}
\qquad\Longrightarrow\qquad
\underbrace{d(u,v) = \|u-v\|}_{\textbf{metric (distance)}}.
$$

So $(V,\|\cdot\|)$ is a **normed space** and $(V,d(\cdot,\cdot))$ is a **metric space**. One object — the inner product — hands you length, distance, and angle all at once. (The angle $\theta$ between $u,v$ comes from $\cos\theta = \langle u,v\rangle/(\|u\|\,\|v\|)$.)

## 1.4 Three inner products you must recognize

The same abstract definition covers finite vectors **and** functions. That unity is the reason kernels work.

**Example 1 — the space $\mathbb{R}^d$** with the ordinary dot product:

$$
\langle u,v\rangle = u^{\top}v = \sum_{i=1}^d u_i v_i.
$$

This induces the familiar Euclidean ($L_2$) norm

$$
\|v\|_2 = \Big(\sum_{i=1}^d v_i^2\Big)^{1/2}.
$$

**Example 2 — the function space $L^2([a,b])$** with the "continuous dot product," where the sum over coordinates becomes an integral:

$$
\langle f,g\rangle = \int_a^b f(x)\,g(x)\,dx,
\qquad
\|f\|_2 = \Big(\int_a^b f(x)^2\,dx\Big)^{1/2}.
$$

> **See the analogy.** A vector in $\mathbb{R}^d$ has components $v_1,\dots,v_d$; the dot product multiplies matching components and sums. A function has a "component" $f(x)$ at every point $x$; the inner product multiplies matching values and integrates. **Same idea, one has finitely many components, the other infinitely many.**

**Example 3 — the Sobolev space $H^1([a,b])$**, functions whose derivative is also square-integrable, $H^1=\{f\in L^2([a,b]): f'\in L^2([a,b])\}$, with the **Sobolev inner product**

$$
\langle f,g\rangle = \int_a^b f(x)g(x)\,dx + \int_a^b f'(x)g'(x)\,dx.
$$

This one also measures agreement of *slopes*, not just values — a hint that inner products can encode "smoothness," which is exactly what a good regularizer wants.

---

# Part 2 — Reproducing Kernel Hilbert Spaces (RKHS)

## 2.1 The mystery we're solving

We keep seeing four objects floating around, and their relationship is still fuzzy:

- the **input space** $\mathcal{X}$,
- a **feature space** $\mathcal{F}$,
- a **feature map** $\phi:\mathcal{X}\to\mathcal{F}$,
- a **kernel** $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$, and the inner product $\langle\phi(x_i),\phi(x_j)\rangle_{\mathcal{F}}$.

We *want* the clean identity

$$
\boxed{\,k(x,x') = \langle \phi(x),\phi(x')\rangle_{\mathcal{F}}\quad\forall x,x'\in\mathcal{X}\,}
$$

— "the kernel is the inner product of features." The goal of this Part is to show **where $\phi$ and $k$ come from** and why this identity holds automatically. Let's explore.

## 2.2 Hilbert spaces = inner-product spaces with no holes

We build $\mathcal{F}$ as a Hilbert space of functions. A **Hilbert space** is an inner-product space that is also **complete**.

> **Intuition for completeness.** "No missing points." If a sequence of vectors is bunching up (its elements get arbitrarily close to each other), then the limit it's converging to is actually *in the space*. Rational numbers are **not** complete ($1,1.4,1.41,1.414,\dots$ bunches up toward $\sqrt2$, which is missing). Real numbers **are**. Completeness is what lets us safely take limits — and hence guarantee that minimizers of our loss functions exist.

So $(\mathcal{F},\langle\cdot,\cdot\rangle_{\mathcal{F}})$ being a Hilbert space over $\mathbb{R}$ means: it's a vector space (we can add and scale), it has an inner product (we can measure angles, lengths, distances), and it is complete (no missing limits). Formally, completeness says: for every sequence $\{f_i\}_{i=1}^{\infty}$ in $\mathcal{F}$ that is **Cauchy** — for every $r>0$ there is an $N\in\mathbb{N}$ such that $d(f_n,f_m)_{\mathcal{F}} < r$ for all $m,n>N$ — the sequence has a limit that also lies in $\mathcal{F}$.

**Now the crucial extra condition.** Let $(\mathcal{F},\langle\cdot,\cdot\rangle_{\mathcal{F}})$ be a Hilbert space whose **elements are functions** $f:\mathcal{X}\to\mathbb{R}$ (so $s_1 f_1 + s_2 f_2$, $\langle f_1,f_2\rangle_{\mathcal{F}}$ and $\|f_1\|_{\mathcal{F}}$ all make sense). Define, for a fixed input point $x$, the **evaluation functional**

$$
\operatorname{eval}_x:\mathcal{F}\to\mathbb{R},\qquad \operatorname{eval}_x(f) = f(x),
$$

which just reports the value of a function at $x$. We call $\mathcal{F}$ a **Reproducing Kernel Hilbert Space (RKHS)** if $\operatorname{eval}_x$ is **continuous** for every $x\in\mathcal{X}$.

Continuity at $f$ means: for all $\varepsilon>0$ there exists $\delta>0$ such that

$$
d_{\mathcal{F}}(f',f)<\delta \ \Longrightarrow\ |\operatorname{eval}_x(f')-\operatorname{eval}_x(f)| = |f'(x)-f(x)| < \varepsilon.
$$

> **What this condition buys us.** It says: *if two functions are close in the space's norm, then their values at every point are close.* In other words, "being nearby as functions" implies "agreeing pointwise." That well-behavedness is precisely what makes pointwise evaluation — the thing we do when we make a prediction $f(x)$ — trustworthy.

## 2.3 The reproducing property (this is where the kernel is born)

We now pull the kernel out of thin air using one classical theorem.

**Riesz Representation Theorem.** In a Hilbert space $\mathcal{F}$, every **continuous linear functional** $\psi\in\mathcal{F}^{*}$ can be written as an inner product against a *single fixed vector*: there exists a unique $f_\psi\in\mathcal{F}$ with

$$
\psi[f] = \langle f, f_\psi\rangle_{\mathcal{F}}\qquad \forall f\in\mathcal{F}.
$$

Here $\mathcal{F}^{*}$ is the **dual space** of $\mathcal{F}$ — the space of all linear and continuous maps from $\mathcal{F}$ to $\mathbb{R}$.

> **Intuition.** Every "linear measurement" you can perform on a Hilbert space is secretly a dot product with some fixed vector. The measurement device *is* a vector.

Apply this to $\operatorname{eval}_x$. It is linear (evaluating $a f + b g$ at $x$ gives $a f(x)+b g(x)$), and in an RKHS it is continuous — so Riesz applies. There must exist a special element, call it $k_x\in\mathcal{F}$, such that

$$
\boxed{\,f(x) = \operatorname{eval}_x(f) = \langle f,\,k_x\rangle_{\mathcal{F}}\qquad\forall f\in\mathcal{F}.\,}
$$

This is the **reproducing property**: taking the inner product of any function $f$ with the magic element $k_x$ "reproduces" the value $f(x)$.

Now here's the move. The representer $k_x$ is *itself a function* $\mathcal{X}\to\mathbb{R}$ (it lives in $\mathcal{F}$, and $\mathcal{F}$ is a space of functions). So we can evaluate **it** at another point $x'$. Using the reproducing property again, with $f=k_x$:

$$
k_x(x') = \operatorname{eval}_{x'}(k_x) = \langle k_x,\,k_{x'}\rangle_{\mathcal{F}}.
$$

Define the **kernel** and the **feature map** by

$$
k(x,x') := k_x(x') = \langle k_x,k_{x'}\rangle_{\mathcal{F}},
\qquad
\phi(x) := k_x .
$$

Substituting one into the other gives exactly the identity we wanted:

$$
\boxed{\,k(x,x') = \langle \phi(x),\phi(x')\rangle_{\mathcal{F}}.\,}
$$

**Mystery solved.** The feature map is "map a point $x$ to its evaluation-representer function $k_x$," and the kernel is "the inner product of two such representers." You never chose $\phi$ and $k$ separately — the Hilbert space handed them to you together.

## 2.4 The reproducing kernel is symmetric and positive-definite

Two properties fall out for free, and they matter enormously.

**Symmetry** is immediate from symmetry of the inner product:

$$
k(x,x') = \langle k_x,k_{x'}\rangle_{\mathcal{F}} = \langle k_{x'},k_x\rangle_{\mathcal{F}} = k(x',x).
$$

**Positive-definiteness.** For any finite set of points $\{x_i\}_{i=1}^n$ and any coefficients $\{c_i\}_{i=1}^n\subset\mathbb{R}$,

$$
\sum_{i,j} c_i c_j\, k(x_i,x_j)
= \sum_{i,j} c_i c_j\, \langle k_{x_i},k_{x_j}\rangle_{\mathcal{F}}
= \Big\langle \sum_i c_i k_{x_i},\ \sum_j c_j k_{x_j}\Big\rangle_{\mathcal{F}}
= \Big\|\sum_i c_i k_{x_i}\Big\|_{\mathcal{F}}^2 \ \ge\ 0.
$$

The middle step is just linearity of the inner product; the last is the definition of the norm. A squared length can't be negative — so the kernel is positive-definite.

> **Why we care.** These two properties are the entire "admission ticket." Any function $k$ that is symmetric and positive-definite is a legal kernel, and (next section) automatically comes with its own hidden feature space. Positive-definiteness is also what guarantees the kernel matrix $K$ is invertible-friendly, which we'll use when we solve for $\alpha$.

**Key take-away (direction 1).**

> For **every RKHS** $(\mathcal{F},\langle\cdot,\cdot\rangle_{\mathcal{F}})$ we can construct a **unique symmetric, positive-definite kernel** $k_{\mathcal{F}}$. It is called the **reproducing kernel** of that space.

**A concrete example.** Take the dual space of $\mathbb{R}^d$: $(\mathbb{R}^d)^{*}=\{f_\alpha:\mathbb{R}^d\to\mathbb{R},\ x\mapsto \alpha^{\top}x\}$ — all linear maps — with inner product $\langle f_\alpha,f_\beta\rangle_{(\mathbb{R}^d)^{*}} = \langle\alpha,\beta\rangle_{\mathbb{R}^d}$. For a point $x$, the Riesz representer of $\operatorname{eval}_x$ is $f_x$, because

$$
f_\alpha(x) = \langle f_\alpha, f_x\rangle_{(\mathbb{R}^d)^{*}} = \alpha^{\top}x \qquad\text{for all } f_\alpha.
$$

Then, for another point $x'$,

$$
f_x(x') = \langle f_x, f_{x'}\rangle_{(\mathbb{R}^d)^{*}} = x^{\top}x'.
$$

So the reproducing kernel of $(\mathbb{R}^d)^{*}$ is the **linear kernel** $k(x,x')=x^{\top}x'$. The plainest kernel you know is just an RKHS in disguise.

## 2.5 Moore–Aronszajn: the two-way street

Direction 1 said *RKHS $\Rightarrow$ kernel*. In practice we usually want to go the **other way**: pick a convenient similarity function and get a powerful feature space for free. Constructing an RKHS by hand to get a "useful" kernel is hard. Fortunately we don't have to.

**Moore–Aronszajn Theorem.** If $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is a **symmetric, positive-definite** kernel, then there exists a **unique** RKHS $(\mathcal{F},\langle\cdot,\cdot\rangle_{\mathcal{F}})$ whose reproducing kernel is $k$.

Putting both directions together gives a perfect correspondence:

$$
\boxed{\ \text{symmetric positive-definite kernel } k \quad\Longleftrightarrow\quad \text{a unique RKHS } \mathcal{F}\ }
$$

> **This is the whole magic of kernel methods.** To work in some rich, possibly infinite-dimensional feature space $\mathcal{F}$, you do **not** build $\mathcal{F}$, you do **not** compute $\phi(x)$. You just write down a symmetric positive-definite function $k(x,x')$. The space exists, the features exist, and every inner product you'll ever need is just a call to $k$. This is the **kernel trick**.

## 2.6 Kernels you'll actually use

The slides prove the theory with the linear kernel; in practice you pick from a small menu. All are symmetric and positive-definite, so each secretly names an RKHS.

| Kernel | Formula | Feature space |
|---|---|---|
| Linear | $k(x,x')=x^{\top}x'$ | $\mathbb{R}^d$ itself |
| Polynomial (degree $p$) | $k(x,x')=(x^{\top}x'+c)^{p}$ | all monomials up to degree $p$ |
| RBF / Gaussian | $k(x,x')=\exp\!\big(-\|x-x'\|^2/2\sigma^2\big)$ | **infinite**-dimensional |

![RBF kernel](figures/rbf_basis.png)
*Left: the RBF (Gaussian) kernel as a similarity score — it's $1$ when $x=x'$ and decays smoothly with distance; the bandwidth $\sigma$ sets how quickly. Right: place one kernel "bump" $k(\cdot,x_i)$ on each data point and take a weighted sum $f(x)=\sum_i\alpha_i k(x,x_i)$. Any function in the RBF-RKHS looks like this. Hold onto this picture — the representer theorem is about to tell us the best predictor always has exactly this form.*

---

# Part 3 — The Representer Theorem

We now have the arena (RKHS) and its currency (kernels). Time to learn. But a function space can be infinite-dimensional — how could we possibly search it for the best predictor? The representer theorem is the answer, and it is the theoretical heart of the whole course.

## 3.1 Risk, empirical risk, and the objective

Consider a family of **hypotheses** $\mathcal{H}=\{f:\mathcal{X}\to\mathbb{R}\}$ (for linear regression, $\mathcal{H}=\{f_w: x\mapsto w^{\top}x\}$). The **risk** of a hypothesis is its *expected* loss over the true data distribution $p(x,y)$:

$$
\mathcal{R}[f] = \mathbb{E}_{(x,y)\sim p(x,y)}\big[\mathcal{L}(f(x),y)\big],
\qquad \mathcal{L}:\mathbb{R}\times\mathbb{R}\to\mathbb{R}_{\ge 0}.
$$

The dream is $f^{*}=\arg\min_{f\in\mathcal{H}}\mathcal{R}[f]$. The problem: $p(x,y)$ is **unknown** — we only have samples. So we assume our data is drawn i.i.d., $\{(x_i,y_i)\}_{i=1}^{n}\overset{\text{i.i.d.}}{\sim}p(x,y)$, and minimize the **regularized empirical risk** instead:

$$
\boxed{\,f^{*} = \arg\min_{f\in\mathcal{F}}\ \underbrace{\frac1n\sum_{i=1}^{n}\mathcal{L}(f(x_i),y_i)}_{\text{fit the data}}\ +\ \underbrace{g(\|f\|_{\mathcal{F}})}_{\text{stay simple}}\,}
$$

where $g:[0,\infty)\to\mathbb{R}$ is a **strictly increasing** function of the norm. The first term rewards fitting the training data; the second penalizes "large" (complicated) functions. That second term is regularization — the subject of Part 4 — and it is exactly the ingredient that makes the coming theorem work.

## 3.2 The theorem

> **Representer Theorem.** Given a kernel $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ with RKHS $(\mathcal{F},\langle\cdot,\cdot\rangle_{\mathcal{F}})$, the minimizer $f^{*}$ of the regularized empirical risk in $\mathcal{F}$ has the form
> $$
> f^{*}(\cdot) = \sum_{i=1}^{n}\alpha_i\,k(\cdot,x_i),\qquad \alpha_i\in\mathbb{R}.
> $$

Read that again: the best function in a possibly **infinite-dimensional** space is a **finite** sum — one term per training point. The search collapses from "all of $\mathcal{F}$" to "just the $n$ numbers $\alpha_1,\dots,\alpha_n$." That is why kernel methods are computable at all.

Before the proof, one identity we'll reuse. With the feature map $\phi(x)=k(\cdot,x)$, the reproducing property gives

$$
[\phi(x)](x') = k(x',x) = \langle \phi(x'),\phi(x)\rangle_{\mathcal{F}},
$$

and therefore, writing $f^{*}$ in the boxed form,

$$
f^{*}(x)=\sum_{i=1}^n\alpha_i k(x,x_i)=\sum_{i=1}^n\alpha_i k(x_i,x)=\sum_{i=1}^n\alpha_i\langle\phi(x_i),\phi(x)\rangle_{\mathcal{F}}=\Big\langle \underbrace{\sum_{i=1}^n\alpha_i\phi(x_i)}_{=:v^{*}},\ \phi(x)\Big\rangle_{\mathcal{F}}.
$$

So $f^{*}$ corresponds to a single feature-space vector $v^{*}=\sum_i\alpha_i\phi(x_i)$ living in the span of the training features.

## 3.3 Proof (in three clean steps)

The proof is a beautiful "orthogonality" argument. We show the optimal $f$ can have no component outside the span of the training data.

**Step 1 — Orthogonal decomposition.** We minimize over the RKHS $\mathcal{H}=\mathcal{F}$. Consider the feature map $\phi:\mathcal{X}\to\mathcal{F},\ x\mapsto k(\cdot,x)$, which satisfies $[\phi(x)](x')=k(x',x)=\langle\phi(x'),\phi(x)\rangle_{\mathcal{F}}$. Let

$$
S := \operatorname{span}\big(\{\phi(x_i)\}_{i=1}^{n}\big)
$$

be the (at most $n$-dimensional) subspace spanned by the training features. Any $f\in\mathcal{F}$ splits **uniquely** into a part inside $S$ and a part orthogonal to it:

$$
f = u + v,\qquad u\in S,\quad v\in S^{\perp},\quad \langle u,v\rangle = 0.
$$

![Orthogonal decomposition](figures/orthogonal_decomposition.png)
*The blue plane is $S$, spanned by the training features. Any candidate function $f$ (blue arrow) is the sum of its shadow $u$ inside the plane (green) and a leftover $v$ sticking straight out of it (orange). The proof shows the leftover $v$ never helps — so the best $f$ lies flat in the plane.*

**Step 2 — the fit term ignores $v$.** Evaluating $f$ at a training point $x_j$ uses the reproducing property (a Riesz representation), then orthogonality:

$$
f(x_j)=\langle f,\phi(x_j)\rangle = \langle u+v,\,\phi(x_j)\rangle = \langle u,\phi(x_j)\rangle + \underbrace{\langle v,\phi(x_j)\rangle}_{=\,0\ \text{since }\phi(x_j)\in S,\ v\perp S}.
$$

Since $u\in S$ we can write $u=\sum_{i}\alpha_i\phi(x_i)$, so

$$
f(x_j)=\sum_{i=1}^n\alpha_i\,\langle \phi(x_i),\phi(x_j)\rangle = \sum_{i=1}^n\alpha_i\,k(x_i,x_j).
$$

**The predictions at all training points depend only on $u$ — not on $v$.** Hence the entire data-fit term $\frac1n\sum_j\mathcal{L}(f(x_j),y_j)$ is unchanged if we drop $v$. We may as well assume

$$
f(\cdot)=u(\cdot)=\sum_{i=1}^n\alpha_i[\phi(x_i)](\cdot)=\sum_{i=1}^n\alpha_i\,k(\cdot,x_i).
$$

**Step 3 — the penalty term *punishes* $v$.** Because $u\perp v$, the Pythagorean theorem gives

$$
\|f\| = \|u+v\| = \sqrt{\|u\|^2 + 2\underbrace{\langle u,v\rangle}_{=0} + \|v\|^2} = \sqrt{\|u\|^2+\|v\|^2}\ \ge\ \sqrt{\|u\|^2}=\|u\|,
$$

with equality **iff** $v=0$. Since $g$ is strictly increasing,

$$
g(\|f\|) = g\Big(\sqrt{\|u\|^2+\|v\|^2}\Big)\ \ge\ g\Big(\sqrt{\|u\|^2}\Big)=g(\|u\|).
$$

So any nonzero $v$ *strictly increases* the regularizer while leaving the fit untouched — pure loss.

**Conclusion.** The first term is independent of $v$; the second is minimized by $v=0$. Therefore the minimizer must have $v=0$, giving

$$
f^{*}(\cdot)=u(\cdot)=\sum_{i=1}^{n}\alpha_i\,k(\cdot,x_i).\qquad\blacksquare
$$

> **The one-sentence version.** Any part of a candidate function that sticks out of the span of the training features is invisible to the data but visible to the penalty — so the optimum throws it away, and what remains is a finite kernel expansion.

**Corollary (matrix form).** Stacking the training predictions into a vector, with the **kernel matrix** $K(X,X)$ where $[K(X,X)]_{ij}=k(x_i,x_j)$,

$$
\hat{y} = K(X,X)\,\alpha.
$$

## 3.4 Application: Kernel Ridge Regression (full derivation)

Let's turn the theorem into a concrete, closed-form algorithm — this is the derivation the slides walk through, with every step shown.

**Setup.** Data $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$ with $x_i\in\mathbb{R}^d,\ y_i\in\mathbb{R}$. Use the **squared-error loss** $\mathcal{L}(f(x_i),y_i)=(f(x_i)-y_i)^2$ and the **squared-norm regularizer** $g(\|f\|)=\lambda\|f\|^2$ for some $\lambda>0$:

$$
f^{*}=\arg\min_{f\in\mathcal{F}}\ \frac1n\sum_{i=1}^n (f(x_i)-y_i)^2 + \lambda\|f\|^2 .
$$

**Step 1 — invoke the representer theorem.** We know $f^{*}(\cdot)=\sum_{i=1}^n\alpha_i k(\cdot,x_i)$ for some $\alpha=(\alpha_1,\dots,\alpha_n)^{\top}\in\mathbb{R}^n$. Now we just need the best $\alpha$.

**Step 2 — rewrite everything in terms of $\alpha$ and $K$.** Let $K=K(X,X)$ (an $n\times n$ symmetric matrix). Two facts:

- **Predictions:** $f^{*}(x_j)=\sum_i\alpha_i k(x_j,x_i)=(K\alpha)_j$, so the vector of fitted values is $K\alpha$.
- **Norm:** using $f^{*}=\sum_i\alpha_i\phi(x_i)$,
$$
\|f^{*}\|^2 = \Big\langle \sum_i\alpha_i\phi(x_i),\ \sum_j\alpha_j\phi(x_j)\Big\rangle = \sum_{i,j}\alpha_i\alpha_j\,k(x_i,x_j) = \alpha^{\top}K\alpha .
$$

Substitute into the objective and expand the squared norm (using $K^{\top}=K$):

$$
h(\alpha):=\frac1n\|K\alpha - y\|^2 + \lambda\,\alpha^{\top}K\alpha
= \frac1n\Big[\underbrace{\alpha^{\top}K^2\alpha - 2\alpha^{\top}Ky + y^{\top}y}_{\|K\alpha-y\|^2}\Big] + \lambda\,\alpha^{\top}K\alpha .
$$

**Step 3 — take the gradient and set it to zero.** Using $\nabla_\alpha(\alpha^{\top}M\alpha)=2M\alpha$ for symmetric $M$, and $\nabla_\alpha(\alpha^{\top}b)=b$:

$$
\nabla_\alpha h(\alpha) = \frac1n\big[2K^2\alpha - 2Ky\big] + 2\lambda K\alpha \overset{!}{=}0.
$$

Multiply by $\tfrac{n}{2}$ and rearrange:

$$
K^2\alpha - Ky + n\lambda K\alpha = 0
\quad\Longleftrightarrow\quad
K^2\alpha + n\lambda K\alpha = Ky .
$$

Factor a $K$ on the left, $K\big(K\alpha + n\lambda\alpha\big)=Ky$. Assuming $K$ is invertible, cancel it:

$$
(K + n\lambda I)\,\alpha = y
\quad\Longrightarrow\quad
\boxed{\,\alpha^{*} = (K + n\lambda I)^{-1}\,y\,}.
$$

> **A free bonus of regularization.** Even if $K$ is singular (e.g. duplicate points), $K+n\lambda I$ is **always invertible** for $\lambda>0$, because adding $n\lambda I$ shifts every eigenvalue up by $n\lambda>0$. Regularization doesn't just fight overfitting — it makes the linear system solvable.

**Step 4 — predict at a new point $x$.** Plug $\alpha^{*}$ back into the kernel expansion. With the row vector $K(x,X)=\big(k(x,x_1),\dots,k(x,x_n)\big)\in\mathbb{R}^n$,

$$
\boxed{\,f^{*}(x)=K(x,X)\,\alpha^{*}=K(x,X)\,(K(X,X)+n\lambda I)^{-1}\,y\,}.
$$

![Kernel ridge fit](figures/kernel_ridge_fit.png)
*Kernel ridge regression on noisy data with an RBF kernel. The faint grey curves are the individual weighted bumps $\alpha_i k(\cdot,x_i)$; the bold curve is their sum $f^{*}$. Exactly the shape the representer theorem promised — one bump per data point, combined with the learned weights $\alpha^{*}$.*

**Notice what never appeared:** the feature vectors $\phi(x_i)$, the dimension $p$ of feature space, or $w$. Every quantity is expressed through kernel evaluations only. That is the kernel trick paying off — we optimized in a possibly infinite-dimensional space using an $n\times n$ system.

## 3.5 Cost: input space vs. feature space

Why kernelize? Compare the ways to solve (regularized) linear regression. Let $X\in\mathbb{R}^{n\times d}$ be the data in input space and $Z\in\mathbb{R}^{n\times p}$ the data in an explicit feature space of dimension $p$.

| Method | Solution | Cost |
|---|---|---|
| Linear regression in $\mathbb{R}^d$ | $w^{*}=(X^{\top}X)^{-1}X^{\top}y$ | $\mathcal{O}(d^2 n + d^3)$ |
| Linear regression in feature space $\mathcal{F}$ | $w^{*}=(Z^{\top}Z)^{-1}Z^{\top}y$ | $\mathcal{O}(p^2 n + p^3)$ |
| **Kernel** regression | $\alpha^{*}=K^{-1}y$, then $\hat y = K\alpha$ | $\mathcal{O}(n^2 d + n^3)$ |

> **The trade.** Explicit features cost $\mathcal{O}(p^3)$ — hopeless when $p$ is huge or infinite. The kernel form costs $\mathcal{O}(n^3)$, **independent of $p$**. So kernels win big when the feature space is enormous but the dataset is moderate ($n \ll p$). The flip side: when $n$ is very large, the $n\times n$ kernel matrix becomes the bottleneck — that's when you'd go back to explicit features or approximations. In the kernel view we compute $\alpha^{*}$ directly and the explicit weight vector $w^{*}$ is simply **avoided**.

**Where does the finite expansion come from without an RKHS?** Even plain gradient descent gives a hint. If we ever represent the weights as a linear combination of the training features,

$$
w=\sum_{i=1}^n\alpha_i\phi(x_i)=Z^{\top}\alpha,\qquad Z=(\phi(x_1),\dots,\phi(x_n))^{\top}\in\mathbb{R}^{n\times p},
$$

then any update of the form $g(Zw)$ becomes $g(ZZ^{\top}\alpha)=g\big(K(X,X)\,\alpha\big)$ — again only kernels appear. The representer theorem is the general principle that guarantees this always happens for regularized empirical risk. (For example, an $L_2$-regularized linear model $\mathcal{L}(w)=\sum_i \ell_i(w^{\top}x_i,y_i)+\tfrac{\lambda}{2}\|w\|^2$ has optimum $w^{*}=\sum_i \alpha_i x_i = X^{\top}\alpha$ with $\alpha_i=-\lambda^{-1}\ell_i'((w^{*})^{\top}x_i,y_i)$ — a linear combination of the training points, exactly as promised.)

---

# Part 4 — Regularization

The representer theorem *required* a regularizer $g(\|f\|)$. Part 4 is about what that term actually does, and the two most important choices for it: $L_2$ and $L_0$.

## 4.1 Why regularize?

Complex models fit training data beautifully and then fail on new data — **overfitting**.

![Polynomial overfitting](figures/poly_overfit.png)
*Fitting polynomials of increasing degree to the same 11 noisy points from a sine curve (green dashed = truth). Degree 1 is too rigid (**underfits**). Degree 3 captures the shape (**good**). Degree 9 threads every point but oscillates wildly between them (**overfits**) — it memorized the noise.*

But we often genuinely **need** complex models: the true relationship between $x_i$ and $y_i$ may require a high-degree polynomial or many interacting features. So we can't just "keep it simple." Our two main tools:

- **Regularization** — add a penalty on model complexity (this Part).
- **Ensembling** — average many models to cut variance (a later topic).

**Which model should you pick?** Suppose two linear fits have the *same* training error:

![Which line](figures/which_line.png)
*A steep line (red, large $\|w\|$) and a shallow line (green, small $\|w\|$) with essentially identical training error. Prefer the green one.*

Why prefer the flatter line? Because a **smaller slope** means a small change in $x_i$ produces only a small change in the prediction $y_i$; the model is **less sensitive to getting $w$ exactly right**, and its test error is typically lower. Large weights make a model twitchy and fragile. Concretely, the same degree-7 curve can be written with tame or wild coefficients:

$$
\hat{y}_i = 0.0001\,x_i^{7} + 0.03\,x_i^{3} + 3
\qquad\text{vs.}\qquad
\hat{y}_i = 1000\,x_i^{7} - 500\,x_i^{6} + 890\,x_i .
$$

Both are degree-7 polynomials; the left one is smooth and stable, the right one is a rollercoaster. **Regularization is the mechanism that steers us toward the tame version by penalizing large weights.**

## 4.2 $L_2$ regularization (ridge)

The standard choice. Add the squared $L_2$-norm of the weights to the squared-error loss:

$$
\mathcal{L}(w)=\frac12\sum_{i=1}^n (w^{\top}x_i - y_i)^2 + \frac{\lambda}{2}\sum_{j=1}^{d} w_j^2
\qquad\Longleftrightarrow\qquad
\mathcal{L}(w)=\frac12\|Xw-y\|^2 + \frac{\lambda}{2}\|w\|^2 .
$$

Here $\lambda>0$ is the **regularization parameter**.

> **The mantra.** Regularized loss balances *low error* against *small weights*: **"you may increase the training error, if doing so makes $w$ smaller."** This nearly always reduces overfitting.

**The fundamental trade-off**, stated crisply:

- Regularization **increases** training error (you're no longer free to chase every point).
- Regularization **decreases** approximation (generalization) error — up to a point.

![Train/test trade-off](figures/train_test.png)
*As $\lambda$ grows (left to right), training error rises monotonically. Test error is U-shaped: too little regularization overfits (high variance), too much underfits (high bias). The sweet spot in the middle is the $\lambda$ we're hunting for.*

The same story, shown as actual fitted curves (here with a kernel model, but the intuition is identical for linear ridge):

![Effect of lambda](figures/lambda_effect.png)
*Left: a tiny $\lambda$ lets the model chase every point — wiggly, overfit. Right: a huge $\lambda$ forces the weights so small the curve goes nearly flat — underfit. Middle: a balanced $\lambda$ recovers the smooth underlying trend.*

**The geometry.** In weight space, the squared-error term has elliptical contours around the unregularized optimum. The penalty $\tfrac{\lambda}{2}\|w\|^2$ pulls the solution toward the origin:

![L2 contour](figures/l2_contour.png)
*Blue ellipses are contours of equal squared error (center = unregularized $w^{*}$, red). Adding the $L_2$ penalty moves the solution to the green point — same ballpark training error, but noticeably smaller weights. The arrow is the shrinkage $L_2$ applies.*

### Solving $L_2$ regression exactly

**Unregularized first.** Expand the objective $\mathcal{L}(w)=\tfrac12\|Xw-y\|^2$:

$$
\mathcal{L}(w)=\frac12 w^{\top}X^{\top}Xw - w^{\top}X^{\top}y + \frac12 y^{\top}y .
$$

Gradient, then set to zero (these are the **normal equations**):

$$
\nabla_w\mathcal{L}(w)=X^{\top}Xw - X^{\top}y \overset{!}{=}0
\ \Longleftrightarrow\ X^{\top}Xw=X^{\top}y
\ \Longrightarrow\ w=(X^{\top}X)^{-1}X^{\top}y \quad(\text{if }X^{\top}X\text{ invertible}).
$$

**Now regularized.** Add $\tfrac{\lambda}{2}\|w\|^2=\tfrac{\lambda}{2}w^{\top}w$. The objective and gradient pick up one term each (shown in red conceptually):

$$
\mathcal{L}(w)=\frac12\|Xw-y\|^2 + \frac{\lambda}{2}w^{\top}w,
\qquad
\nabla_w\mathcal{L}(w)=X^{\top}Xw - X^{\top}y + \lambda w .
$$

Set to zero:

$$
X^{\top}Xw - X^{\top}y + \lambda w = 0
\ \Longleftrightarrow\ (X^{\top}X + \lambda I)\,w = X^{\top}y
\ \Longrightarrow\
\boxed{\,w=(X^{\top}X + \lambda I)^{-1}X^{\top}y\,}.
$$

As noted before, $X^{\top}X + \lambda I$ is **always invertible** for $\lambda>0$. (Compare with kernel ridge in §3.4: same idea, dual variables.)

**By gradient descent** (when $d$ is large and you'd rather not invert a matrix): with the gradient above, the update rule is

$$
w^{(t+1)} \leftarrow w^{(t)} - \gamma\,\nabla_w\mathcal{L}(w) = w^{(t)} - \gamma\big[X^{\top}(Xw^{(t)}-y) + \lambda w^{(t)}\big],
$$

at a cost of only $\mathcal{O}(nd)$ per iteration (the unregularized version drops the $\lambda w^{(t)}$ term).

### What $\lambda$ actually does — numbers

The regularization parameter $\lambda$ is a **hyperparameter** controlling the strength: large $\lambda$ puts a large penalty on the weights, so training error goes up and approximation error goes down. Here is a real ridge solution on a small 5-feature dataset as $\lambda$ sweeps (computed, not sketched):

| $\lambda$ | $w_1$ | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $\|Xw-y\|^2$ | $\|w\|^2$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.98 | −1.40 | 1.06 | 0.45 | −0.78 | 8.77 | 7.81 |
| 1 | 1.89 | −1.31 | 0.99 | 0.48 | −0.74 | 9.14 | 7.06 |
| 4 | 1.67 | −1.12 | 0.83 | 0.53 | −0.64 | 13.04 | 5.42 |
| 16 | 1.17 | −0.73 | 0.48 | 0.53 | −0.43 | 37.92 | 2.59 |
| 64 | 0.56 | −0.32 | 0.16 | 0.34 | −0.18 | 100.40 | 0.59 |
| 256 | 0.19 | −0.10 | 0.04 | 0.12 | −0.05 | 158.69 | 0.07 |
| 1024 | 0.05 | −0.03 | 0.01 | 0.04 | −0.01 | 184.12 | 0.00 |

Read the two rightmost columns: as $\lambda$ grows, the fit error $\|Xw-y\|^2$ **increases** and the weight size $\|w\|^2$ **decreases** — the trade-off made numeric. At $\lambda=0$ we recover ordinary least squares; as $\lambda\to\infty$ every weight is crushed toward $0$. Note individual $w_j$ can move either way, but because we penalize the *squared* norm, the **largest** weights are shrunk the most (squaring makes big weights expensive).

Plotting the weights against $\lambda$ gives the **regularization path**:

![Regularization path](figures/reg_path.png)
*Each curve is one coefficient $w_j$ as $\lambda$ increases. The path starts at the unregularized least-squares solution ($\lambda=0$, left) and every coefficient is smoothly pulled toward $0$ as $\lambda\to\infty$.*

**How to choose $\lambda$?**
- **Theory:** as the sample size $n$ grows, a good $\lambda$ lies roughly in the range $\mathcal{O}(1)$ to $\mathcal{O}(\sqrt{n})$.
- **Practice:** pick the $\lambda$ that minimizes **validation error** (or cross-validation error). This is the U-curve from the trade-off figure, evaluated on held-out data.

## 4.3 $L_0$ regularization: feature selection

$L_2$ shrinks weights but rarely sets them exactly to zero. Sometimes what we really want is to **throw entire features away** — this is **feature selection**, and it corresponds to a different penalty.

**The dilemma.** When we build a model we must decide which features to use:

- **Too few features** → not enough information → bad predictions. (Predicting spam from *only* whether the word "buy" appears.)
- **Too many features** → the model latches onto irrelevant noise → bad predictions. (Predicting spam from the *temperature* on the day the email was sent. Add enough random features and your training error hits zero — while learning nothing.)

Among many candidate features $x_{\cdot j}$, we want to **select** the ones that actually help predict $y$. "Number of features" is itself a hyperparameter: more features → lower training error but higher approximation error.

**Search and score.** The standard framework:

1. Define a score function $\text{score}(S)$ measuring the quality of a feature subset $S$ (e.g. a loss).
2. Search for the subset $S$ with the best score.

For a linear regression model, a natural score is the squared error using only the features in $S$:

$$
\text{score}(S)=\frac12\sum_{i=1}^n\big(w_S^{\top}x_{i\mid S} - y_i\big)^2,
$$

where $x_{i\mid S}$ keeps only the features in $S$ and $w_S$ are the corresponding weights.

**On which data do we score?** *Not the training set* — training error only ever goes **down** as you add features, so scoring on training data always selects **all** features. Scoring on a **validation set** is better (find the subset with lowest validation error, which is what minimizes test error). But there is a subtlety: with $d$ features there are $2^d$ subsets, and searching all of them invites **optimization bias / false positives** — some irrelevant feature will help *by chance* on the validation set.

**The fix: a complexity penalty.** Charge for every feature you include:

$$
\text{score}(S)=\frac12\sum_{i=1}^n\big(w_S^{\top}x_{i\mid S}-y_i\big)^2 + \lambda\cdot\text{size}(S).
$$

Now, if two subsets have similar squared error, the score prefers the **smaller** one. In words: **"you may add a feature, but only if it reduces the training error by at least $\lambda$."** (Setting $w_1=10^{-5}$ has a negligible effect on the error, so it's better to drop that feature entirely.)

**Connecting to a norm.** In a linear model, setting $w_j=0$ is exactly the same as **removing** feature $x_{\cdot j}$:

$$
\hat y_i = w_1 x_{i1}+w_2 x_{i2}+w_3 x_{i3}+\dots
\ \xrightarrow{\ w_2=0\ }\
\hat y_i = w_1 x_{i1}+0\cdot x_{i2}+w_3 x_{i3}+\dots = w_1 x_{i1}+w_3 x_{i3}+\dots
$$

The **$L_0$ "norm"** counts the nonzero entries of a vector, so it equals the number of features used:

$$
\|w\|_0 = \#\{j : w_j\neq 0\} = \text{size}(S).
\qquad
\text{e.g. } (1,0,2,0,3)^{\top}\Rightarrow\|w\|_0=3,\quad (0,0,0,0,0)^{\top}\Rightarrow\|w\|_0=0.
$$

(It's not a true norm — it isn't homogeneous — hence the quotes.) A small $\|w\|_0$ means the model uses few features. Putting it together, feature selection with a complexity penalty **is** $L_0$-regularized regression:

$$
\boxed{\ \mathcal{L}(w)=\underbrace{\frac12\|Xw-y\|^2}_{\text{prediction error}} + \underbrace{\lambda\|w\|_0}_{\text{degrees of freedom}}\ }
$$

Its behavior across $\lambda$:

- $\lambda=0$: no penalty on nonzeros → minimize plain squared loss → use **all** features.
- $\lambda=\infty$: infinite penalty → minimum at $w=0$ → use **no** features.
- $0<\lambda<\infty$: larger $\lambda$ emphasizes zeros → **more features removed**.

![L0 feature count](figures/l0_features.png)
*Training error (blue) drops monotonically as you add features. Validation error (orange) is U-shaped — too few features underfit, too many overfit. $L_0$ regularization, with $\lambda$ tuned, targets the bottom of that U.*

**Optimization is hard.** Finding the best subset means searching $2^d$ possibilities — infeasible. A practical **greedy** method is **forward selection**:

1. Start with empty sets $S=\{\}$ and best-so-far $B=\{\}$.
2. For each feature $x_{\cdot j'}$ not yet in $S$: compute the score of $S\cup x_{\cdot j'}$.
3. Find the feature $x_{\cdot j}$ with the best score in step 2.
4. If $S\cup x_{\cdot j}$ improves on the best score seen so far, set $B\leftarrow S\cup x_{\cdot j}$.
5. If some feature is still not in $S$: set $S\leftarrow S\cup x_{\cdot j}$ and go back to step 2.
6. Return the best feature set $B$.

**Runtime.** Each of the (up to) $d$ steps fits up to $d$ models, so forward selection fits $\mathcal{O}(d^2)$ models instead of $2^d$. Total cost is $\mathcal{O}(d^2)$ times the cost of one fit — for $L_0$-regularized squared loss that's $\mathcal{O}(nd^4+d^5)$, since fitting a single regression model costs $\mathcal{O}(nd^2+d^3)$. Forward selection is **not guaranteed** to find the globally best subset, but fitting far fewer models makes it cheaper, less prone to overfitting, and less prone to false positives.

## 4.4 Beyond the slides: a word on $L_1$ / Lasso

*(Not in the decks, but the question always comes up, so here's the bridge.)* $L_0$ gives true sparsity but is combinatorially hard. $L_2$ is easy but doesn't zero out features. **$L_1$ regularization** (Lasso) is the best-of-both compromise:

$$
\mathcal{L}(w)=\frac12\|Xw-y\|^2 + \lambda\|w\|_1,\qquad \|w\|_1=\sum_j |w_j|.
$$

It is **convex** (so tractable, unlike $L_0$) yet its diamond-shaped constraint has sharp corners on the axes, so the optimum often lands exactly on a corner — driving some $w_j$ to *exactly* zero and thus selecting features automatically. Think of it as a convex relaxation of $L_0$ that keeps sparsity. It slots neatly between the $L_0$ and $L_2$ stories above.

---

# Part 5 — Putting it all together

You now have the full arc, from an abstract similarity function to a regularized, computable learning algorithm.

**Kernel theory.**
- An **RKHS** is a Hilbert space of functions in which pointwise evaluation is continuous — equivalently, a space where inner products correspond to **kernel evaluations**, $k(x,x')=\langle\phi(x),\phi(x')\rangle$.
- The **reproducing kernel** falls out of the Riesz representation theorem: $f(x)=\langle f,k_x\rangle$, and $k(x,x')=\langle k_x,k_{x'}\rangle$ is automatically symmetric and positive-definite.
- **Moore–Aronszajn**: every symmetric positive-definite kernel is the reproducing kernel of a *unique* RKHS. Pick a kernel and you've picked a feature space — the **kernel trick**.

**Representer theorem.**
- It describes the functional form of the minimizer of the regularized empirical risk: a **finite linear combination of kernel evaluations**, $f^{*}(\cdot)=\sum_{i=1}^n\alpha_i k(\cdot,x_i)$.
- It is the theoretical foundation for **kernelizing** linear methods, and it turns an infinite-dimensional search into solving for $n$ numbers.
- **Application:** kernel ridge regression, $\alpha^{*}=(K+n\lambda I)^{-1}y$, with predictions $f^{*}(x)=K(x,X)\,\alpha^{*}$ — computed purely through kernels.

**Regularization.**
- **$L_2$ (ridge):** shrinks weights toward zero for robustness and stability; exact solution via the modified normal equations $w=(X^{\top}X+\lambda I)^{-1}X^{\top}y$; $\lambda$ chosen by (cross-)validation.
- **$L_0$:** counts nonzero weights, i.e. selects relevant features; optimized greedily by **forward selection**.
- Both are instances of the same principle from the representer theorem's $g(\|f\|)$ term: **pay for complexity, and generalization improves.**

> **The single thread through both lectures.** *Choosing a kernel picks a feature space; regularization tells the representer theorem which function in that space to trust; the result is a finite, kernel-only formula you can actually compute.*

---

*Built from the two THWS "Machine Learning" decks (Kernel Theory; Regularization). All figures were generated programmatically with Python/matplotlib for accuracy — no numbers or curves were sketched by hand.*
