# Maximum Likelihood (MLE) & Maximum A-Posteriori (MAP) — A Friendly Tutorial

*Based on the lectures "Regression Variants & MLE/MAP" and "MLE/MAP & Clustering" (A. Lehrmann, THWS). This tutorial keeps the slides' logic but fills in every skipped algebra step so you can learn it from scratch.*

> **About the figures.** Every picture is produced by an accurate Python script in the `figures/` folder, and the rendered `.png` files are shipped alongside. Images are **not** embedded in this file — when the text says *see Figure 1*, open `figures/fig1_coin_likelihood.png` (or run `python3 figures/fig1_coin_likelihood.py` to regenerate it).

---

## 0. Why bother? The big picture

So far we picked loss functions and regularizers because they "seemed reasonable". MLE and MAP explain **where they actually come from**. This one framework:

- justifies the **"just count"** method for estimating probabilities in naïve Bayes;
- reveals that the **squared loss = assuming Gaussian noise**;
- connects **robust regression** to **heavy-tailed** noise;
- justifies the **sigmoid** in logistic regression;
- shows that **regularizers are priors** (and Laplace smoothing is regularization);
- gives a **general recipe** to turn *any* modelling assumption into a loss you can optimize — even for weird targets like "number of likes" or "number of stars in a review".

The punchline you'll reach:

$$
\boxed{\;\text{choice of \textbf{likelihood} } = \text{choice of \textbf{loss}}, \qquad \text{choice of \textbf{prior}} = \text{choice of \textbf{regularizer}}\;}
$$

---

## 1. Warm-up: `min` vs. `arg min` (a tiny but important distinction)

We keep writing $\min$ and $\max$. Two different questions hide in there:

- $\min_w f(w)$ = **the smallest value** the function reaches. Example: $\min_w (w-1)^2 = 0$.
- $\arg\min_w f(w)$ = **the input(s)** that achieve it. Example: $\arg\min_w (w-1)^2 = 1$.

MLE and MAP are always about the second one — we want the *parameter*, not the height of the curve.

**A subtlety:** the minimizer might not be unique, so $\arg\min$ really returns a **set**. For instance $\arg\max_w \cos(w) = \{\,2k\pi : k\in\mathbb{Z}\,\}$. That's why the careful notation is $\hat{w} \in \arg\min_w f(w)$ ("$\hat w$ *is one of* the minimizers"), not $\hat w = \arg\min$. In practice we usually just say "the" solution and move on, but the $\in$ is why.

---

## 2. The likelihood function

### 2.1 Setup: a coin

We flip a coin three times and observe the data $\mathcal{D} = (H, H, T)$. The parameter we want is

$$
w = P(\text{the coin lands heads}), \qquad w \in [0,1].
$$

### 2.2 Definition

The **likelihood** is the probability of the observed data, *viewed as a function of the parameter $w$*:

$$
p(\mathcal{D}\mid w).
$$

Read that carefully. It's the *same* formula as an ordinary probability, but we treat the **data as fixed** (we already saw $H,H,T$) and let **$w$ vary**. It answers: *"if the true head-probability were $w$, how probable would my observed data be?"*

For our coin, the three flips are independent, so we multiply their probabilities:

$$
p(\mathcal{D}\mid w) \;=\; p(H\mid w)\cdot p(H\mid w)\cdot p(T\mid w) \;=\; w\cdot w \cdot (1-w) \;=\; w^2(1-w).
$$

(If the data were continuous, $p$ would be a probability *density* instead of a probability *mass*, but the logic is identical.)

### 2.3 Plug in a few values to get a feel

| assume $w$ | $p(HHT\mid w) = w^2(1-w)$ | interpretation |
|---|---|---|
| $0.50$ | $0.5\cdot0.5\cdot0.5 = 0.125$ | "likelihood of a fair coin" |
| $0.00$ | $0\cdot0\cdot1 = 0$ | impossible — we *did* see heads |
| $0.75$ | $0.75\cdot0.75\cdot0.25 \approx 0.141$ | data is **more** likely here than for a fair coin |

Already we can smell that $w$ somewhere around $0.7$ explains the data best.

---

## 3. Maximum Likelihood Estimation (MLE)

### 3.1 The idea

**Maximum Likelihood** picks the parameter that makes the observed data as probable as possible:

$$
\boxed{\;\hat{w} \in \arg\max_w \; p(\mathcal{D}\mid w)\;}
$$

For the coin we plot $p(\mathcal{D}\mid w) = w^2(1-w)$ over $w\in[0,1]$. (See **Figure 1**, `figures/fig1_coin_likelihood.png`.) Three things worth noticing:

- the curve is **0 at $w=0$ and $w=1$** — because our data contain *both* an $H$ and a $T$, so a coin that is all-tails or all-heads could never have produced it;
- the peak is **not at $w=0.5$** — we saw more heads than tails, so a fair coin isn't the best explanation;
- $p(\mathcal{D}\mid w)$ is a distribution over the **data $\mathcal{D}$**, not over $w$ — the area under this curve is *not* 1, so it is not a probability distribution of $w$.

### 3.2 Solve it with calculus (filling in the slide's "$\hat w = 2/3$")

Maximize $L(w) = w^2(1-w) = w^2 - w^3$. Take the derivative and set it to zero:

$$
\frac{dL}{dw} = 2w - 3w^2 = w(2 - 3w) \stackrel{!}{=} 0
\;\;\Longrightarrow\;\; w = 0 \;\text{ or }\; w = \tfrac{2}{3}.
$$

$w=0$ is the minimum (likelihood 0); the maximum is

$$
\boxed{\;\hat{w} = \tfrac{2}{3}\;}
$$

which is exactly *"heads observed / total flips" = 2/3*. MLE recovered the intuitive answer.

---

## 4. MLE for binary variables → the "counting" rule of naïve Bayes

Now generalize from three flips to a data matrix $X \in \mathbb{R}^{n\times 1}$ of $n$ i.i.d. binary samples $x_i\in\{0,1\}$, with $p(x_i = 1 \mid w) = w$. Because the samples are independent, the likelihood is a **product**:

$$
p(X\mid w) \;=\; \prod_{i=1}^{n} w^{\,x_i}\,(1-w)^{\,1-x_i}.
$$

*(Check the exponent trick: if $x_i=1$ the term is $w^1(1-w)^0 = w$; if $x_i=0$ it's $w^0(1-w)^1 = 1-w$. Neat.)*

### 4.1 Derivation of the MLE (this step is skipped on the slide)

Products are painful to differentiate, so take the **log** first (allowed — see §5 for why it doesn't move the maximum):

$$
\log p(X\mid w) \;=\; \sum_{i=1}^n \Big[\, x_i \log w + (1-x_i)\log(1-w) \,\Big].
$$

Differentiate w.r.t. $w$ and set to zero. Using $\frac{d}{dw}\log w = \frac1w$ and $\frac{d}{dw}\log(1-w) = \frac{-1}{1-w}$:

$$
\frac{d}{dw}\log p(X\mid w) \;=\; \sum_{i=1}^n \Big[\, \frac{x_i}{w} - \frac{1-x_i}{1-w}\,\Big]
\;=\; \frac{\sum_i x_i}{w} - \frac{n - \sum_i x_i}{1-w} \stackrel{!}{=} 0.
$$

Let $s = \sum_i x_i$ (the number of ones). Multiply through by $w(1-w)$:

$$
s(1-w) - (n-s)\,w = 0
\;\;\Longrightarrow\;\; s - sw - nw + sw = 0
\;\;\Longrightarrow\;\; s = n w.
$$

Therefore:

$$
\boxed{\;\hat{w} = \frac{s}{n} = \frac{\#\text{ of ones}}{\#\text{ of samples}} = \frac{1}{n}\sum_{i=1}^n x_i\;}
$$

**So MLE = "just count the fraction of ones".** This is *exactly* the estimate for the marginal probabilities we plugged into **naïve Bayes** (e.g. $p(y_i=1) = \frac1n\sum_i y_i$). We were secretly doing maximum likelihood all along.

---

## 5. From products to sums: the Negative Log-Likelihood (NLL)

### 5.1 Why take the log at all?

Two reasons, both important:

**(a) It doesn't move the maximum.** The logarithm is **strictly increasing**: if $\alpha > \beta$ then $\log\alpha > \log\beta$. A monotonic transform can stretch a curve but cannot change *where* its highest point is. So

$$
\arg\max_w \, p(\mathcal{D}\mid w) \;=\; \arg\max_w \, \log p(\mathcal{D}\mid w).
$$

Adding a **minus** sign flips "maximize" into "minimize". Combining these, we define the **Negative Log-Likelihood (NLL)** and use it as our objective:

$$
\hat{w} \in \arg\max_w \, p(\mathcal{D}\mid w) \;\equiv\; \arg\min_w \; \underbrace{-\log p(\mathcal{D}\mid w)}_{\text{NLL}(w)}.
$$

(See **Figure 2**, `figures/fig2_log_monotonic.png`: the likelihood and the log-likelihood peak at the *same* $\hat w = 2/3$, and the NLL bottoms out there.)

**(b) It turns products into sums.** This is the practical superpower. The magic identity is

$$
\log\Big(\prod_{i=1}^n \alpha_i\Big) \;=\; \sum_{i=1}^n \log(\alpha_i).
$$

Likelihoods of i.i.d. data are big **products** (and products of many small numbers underflow to zero numerically). Taking the log converts them into **sums**, which are easy to differentiate and numerically stable.

### 5.2 The general i.i.d. MLE

For i.i.d. data $\mathcal{D}$ with $p(\mathcal{D}\mid w) = \prod_{i=1}^n p(\mathcal{D}_i\mid w)$, the MLE is:

$$
\boxed{\;\hat{w} \in \arg\max_w \prod_{i=1}^n p(\mathcal{D}_i\mid w)
\;\equiv\; \arg\min_w \; -\sum_{i=1}^n \log p(\mathcal{D}_i\mid w)\;}
$$

That right-hand side — *minimize a sum of negative log-likelihoods* — should look suspiciously like *minimize a sum of losses*. That's the bridge we build next.

---

## 6. Loss minimization **is** maximum likelihood

Most loss functions we use are secretly an NLL under some noise assumption. We show three: squared, absolute, and logistic.

### 6.1 Squared loss ⟺ Gaussian noise (full derivation)

**The model.** Assume the target is a linear function plus independent Gaussian noise:

$$
y_i = w^\top x_i + \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0,1).
$$

The standard normal density of the noise is

$$
p(\epsilon_i) = \frac{1}{\sqrt{2\pi}}\exp\!\Big(-\frac{\epsilon_i^2}{2}\Big).
$$

*(The lecture slide writes the constant as $\tfrac{1}{2\pi}$; the correct normalizer is $\tfrac{1}{\sqrt{2\pi}}$. It does not matter for the argmin — any constant just adds a constant to the NLL — but here's the right version.)*

**Induced distribution over $y_i$.** Since $\epsilon_i = y_i - w^\top x_i$ is just $y_i$ shifted by the constant $w^\top x_i$, the noise distribution becomes a Gaussian **centered at $w^\top x_i$**:

$$
p(y_i \mid x_i, w) = \frac{1}{\sqrt{2\pi}}\exp\!\Big(-\frac{(w^\top x_i - y_i)^2}{2}\Big) = \mathcal{N}\big(w^\top x_i,\; 1\big).
$$

*(Rule used: if $X\sim\mathcal{N}(\mu,\sigma^2)$ and $Y=aX+b$, then $Y\sim\mathcal{N}(a\mu+b,\,a^2\sigma^2)$. Here $\mu=0,\sigma^2=1,a=1,b=w^\top x_i$.)* Viewed as a function of $w$, this is a **Gaussian likelihood**. (See **Figure 3**, `figures/fig3_gaussian_errors.png`: fit a line, collect the residuals, and their histogram looks Gaussian — that's the assumption made visible.)

**Now minimize the NLL.** Plug the Gaussian likelihood into the i.i.d. NLL and simplify step by step:

$$
\begin{aligned}
\text{NLL}(w) &= -\log p(y\mid X,w) = -\sum_{i=1}^n \log p(y_i\mid x_i, w) \\[1mm]
&= -\sum_{i=1}^n \log\!\left[\frac{1}{\sqrt{2\pi}}\exp\!\Big(-\frac{(w^\top x_i - y_i)^2}{2}\Big)\right] \\[1mm]
&= -\sum_{i=1}^n \left[\underbrace{\log\tfrac{1}{\sqrt{2\pi}}}_{\text{const}} \;-\; \frac{(w^\top x_i - y_i)^2}{2}\right] \qquad (\log(ab)=\log a + \log b,\ \log e^z = z)\\[1mm]
&= \frac{1}{2}\sum_{i=1}^n (w^\top x_i - y_i)^2 \;+\; \text{const} \\[1mm]
&= \frac{1}{2}\lVert Xw - y\rVert^2 + \text{const}.
\end{aligned}
$$

The constant doesn't depend on $w$, so it doesn't affect the $\arg\min$. We are left with

$$
\boxed{\;\text{minimizing the squared loss } \tfrac12\lVert Xw-y\rVert^2 \;=\; \text{MLE under Gaussian noise}\;}
$$

*(Fun fact: this is one reason the linear-regression equations $X^\top X w = X^\top y$ are called the **normal** equations — "normal" as in the normal/Gaussian distribution.)*

### 6.2 Absolute loss ⟺ Laplace noise

Repeat the same argument with a **Laplace** density instead of a Gaussian:

$$
p(y_i\mid x_i, w) = \frac{1}{2}\exp\!\big(-|w^\top x_i - y_i|\big)
\;\;\Longrightarrow\;\;
\text{NLL} = \sum_i |w^\top x_i - y_i| + \text{const} = \lVert Xw - y\rVert_1 + \text{const}.
$$

So the **absolute (robust) loss = MLE under Laplace noise**. And here's the beautiful connection back to Tutorial 1: why is the absolute loss *robust*? Because the Laplace distribution has **heavier tails** than the Gaussian — it assigns more probability to large residuals, so a big outlier is "less surprising" and therefore pulls the fit less. (See **Figure 4**, `figures/fig4_loss_from_likelihood.png`: left = the two densities, right = their negative logs, which *are* the squared and absolute losses.)

The general takeaway: **choosing a loss = choosing a noise distribution.** There are many more such pairs.

### 6.3 Logistic loss ⟺ sigmoid likelihood (full derivation)

For **classification** with labels $y_i \in \{-1, +1\}$, we need a model that outputs a probability. The linear score $z_i = w^\top x_i$ lives in $(-\infty, +\infty)$, so we squash it into $(0,1)$ with the **sigmoid**:

$$
\sigma(z_i) = \frac{1}{1 + \exp(-z_i)} \in (0,1).
$$

(See **Figure 5**, `figures/fig5_sigmoid_logistic.png`, left panel.) Define the probability of the **positive** class as $p(y_i=+1\mid z_i) = \sigma(z_i)$. Then the negative class is

$$
p(y_i = -1\mid z_i) = 1 - \sigma(z_i).
$$

**A slick simplification** (this is the algebra the slide walks through). Compute $1-\sigma(z_i)$:

$$
1 - \sigma(z_i) = 1 - \frac{1}{1+\exp(-z_i)} = \frac{(1+\exp(-z_i)) - 1}{1+\exp(-z_i)} = \frac{\exp(-z_i)}{1+\exp(-z_i)}.
$$

Multiply numerator and denominator by $\exp(z_i)$:

$$
1 - \sigma(z_i) = \frac{\exp(-z_i)\exp(z_i)}{(1+\exp(-z_i))\exp(z_i)} = \frac{1}{\exp(z_i)+1} = \frac{1}{1+\exp(z_i)} = \sigma(-z_i).
$$

So both classes collapse into **one tidy formula** using the label as a sign:

$$
p(y_i \mid z_i) = \sigma(y_i\, z_i)
\qquad\Longrightarrow\qquad
p(y_i \mid x_i, w) = \sigma(y_i\, w^\top x_i) = \big(1 + \exp(-y_i\, w^\top x_i)\big)^{-1}.
$$

*(Check: $y_i=+1$ gives $\sigma(z_i)$; $y_i=-1$ gives $\sigma(-z_i)=1-\sigma(z_i)$. ✓)*

**Now the NLL.** For i.i.d. data,

$$
\begin{aligned}
\text{NLL}(w) &= -\sum_{i=1}^n \log p(y_i\mid x_i,w)
= -\sum_{i=1}^n \log \frac{1}{1+\exp(-y_i w^\top x_i)} \\
&= -\sum_{i=1}^n \Big[\underbrace{\log 1}_{=0} - \log\big(1+\exp(-y_i w^\top x_i)\big)\Big]
= \sum_{i=1}^n \log\!\big(1 + \exp(-y_i\, w^\top x_i)\big).
\end{aligned}
$$

That last expression **is the logistic loss**. So:

$$
\boxed{\;\text{minimizing the logistic loss} \;=\; \text{MLE under a sigmoid likelihood}\;}
$$

This gives logistic regression two equally valid readings:

- a **smooth, convex approximation to the 0-1 loss** (the loss-minimization view — see Figure 5, right panel), and
- a **maximum-likelihood estimate under a sigmoid model** (the probabilistic view).

Training and prediction are unchanged (still minimize the logistic loss in $w$), but the MLE view is what *justifies* interpreting $\sigma(w^\top x)$ as "the probability that this e-mail is important".

---

## 7. Maximum A-Posteriori (MAP) estimation

### 7.1 Two things that bug us about MLE

1. **It's conceptually backwards.** MLE says: *"find the $w$ that makes the data most probable."* But we don't really care how probable the data is — we already have the data. What we actually want is the $w$ that is most probable **given the data**.
2. **It overfits.** Data can be very likely under a **very implausible** $w$ — e.g. a hugely complex model that memorizes the training set. MLE has no way to say "that $w$ is absurd, ignore it".

What we really want:

$$
\text{"find the } w \text{ that has the highest probability } \textbf{given the data } \mathcal{D}\text{."}
$$

### 7.2 The MAP estimate and Bayes' rule

**Maximum A-Posteriori** maximizes the **reverse** conditional — the *posterior* $p(w\mid\mathcal{D})$:

$$
\boxed{\;\hat{w} \in \arg\max_w \; p(w\mid \mathcal{D})\;}
$$

MLE and MAP are linked by **Bayes' rule**:

$$
p(w\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid w)\,p(w)}{p(\mathcal{D})} \;\;\propto\;\; \underbrace{p(\mathcal{D}\mid w)}_{\text{likelihood}}\cdot\underbrace{p(w)}_{\text{prior}}.
$$

We can drop the denominator $p(\mathcal{D})$ because it does **not depend on $w$** (it's a constant as far as the $\arg\max$ is concerned). So MAP maximizes **likelihood × prior**, where:

- the **prior** $p(w)$ encodes our belief about which $w$ are plausible *before* seeing any data;
- e.g. a prior can say "extreme/complex $w$ are unlikely", which is precisely a defence against overfitting.

(See **Figure 6**, `figures/fig6_map_posterior.png`: the likelihood peaks at $\hat w_{\text{MLE}}$, the prior peaks at 0, and their product — the posterior — peaks at $\hat w_{\text{MAP}}$, pulled *toward* zero. That pull is regularization.)

### 7.3 MAP = loss + regularizer (the derivation)

Take the i.i.d. MAP objective and, as always, minimize the negative log:

$$
\begin{aligned}
\hat{w} &\in \arg\max_w \Big[\prod_{i=1}^n p(\mathcal{D}_i\mid w)\Big]\, p(w) \\
&\equiv \arg\min_w \; -\log\Big(\prod_{i=1}^n p(\mathcal{D}_i\mid w)\Big) - \log p(w) \\
&\equiv \arg\min_w \; \underbrace{-\sum_{i=1}^n \log p(\mathcal{D}_i\mid w)}_{\textbf{loss}} \;\; \underbrace{-\,\log p(w)}_{\textbf{regularizer}}.
\end{aligned}
$$

Look at what fell out. The negative log-likelihood is our **loss**; the **negative log-prior is a regularizer**:

$$
\boxed{\;\text{regularizer} = -\log(\text{prior})\;}
$$

Many regularizers you've used are exactly negative log-priors in disguise.

### 7.4 The $L_2$ regularizer is a Gaussian prior (full derivation)

Assume each weight independently has a zero-mean Gaussian prior with variance $\lambda^{-1}$ (so **precision** $\lambda$; a bigger $\lambda$ = a tighter prior = a stronger belief that weights are small):

$$
w_j \sim \mathcal{N}(0, \lambda^{-1}), \qquad
p(w) = \prod_{j=1}^{d} p(w_j) \;\propto\; \prod_{j=1}^d \exp\!\Big(-\frac{\lambda}{2}w_j^2\Big) = \exp\!\Big(-\frac{\lambda}{2}\sum_{j=1}^d w_j^2\Big).
$$

The negative log-prior is then

$$
-\log p(w) = -\log\!\Big(\exp\big(-\tfrac{\lambda}{2}\lVert w\rVert^2\big)\Big) + \text{const} = \frac{\lambda}{2}\lVert w\rVert^2 + \text{const}.
$$

That's the **$L_2$ (ridge) penalty**! So a Gaussian prior on the weights *is* $L_2$ regularization. (See **Figure 7**, `figures/fig7_prior_as_regularizer.png`, left column.)

### 7.5 Mix and match likelihoods and priors

Now combine any likelihood (= loss) with any prior (= regularizer). Two examples from the slides:

**Gaussian likelihood + Gaussian prior → $L_2$-regularized least squares.**

$$
p(y_i\mid x_i,w)\propto\exp\!\Big(-\tfrac{(w^\top x_i - y_i)^2}{2}\Big),\quad
p(w_j)\propto\exp\!\Big(-\tfrac{\lambda}{2}w_j^2\Big)
\;\;\Longrightarrow\;\;
\mathcal{L}(w) = \tfrac12\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2.
$$

That is **ridge regression** — a probabilistic derivation of the exact objective from Tutorial 1 §1.

**Laplace likelihood + Gaussian prior → $L_2$-regularized robust regression.**

$$
p(y_i\mid x_i,w)\propto\exp\!\big(-|w^\top x_i - y_i|\big),\quad
p(w_j)\propto\exp\!\Big(-\tfrac{\lambda}{2}w_j^2\Big)
\;\;\Longrightarrow\;\;
\mathcal{L}(w) = \lVert Xw - y\rVert_1 + \tfrac{\lambda}{2}\lVert w\rVert^2.
$$

And of course a **Laplace prior** on the weights gives the **$L_1$/LASSO penalty** $\lambda\lVert w\rVert_1$ (Figure 7, right column) — closing the loop with the sparsity story of Tutorial 1.

### 7.6 Two subtleties about MAP

- **The prior fades as data grows.** The loss is a **sum over $n$ data points**, while the regularizer is a **single** term $-\log p(w)$ that does *not* grow with $n$. So as $n\to\infty$ the loss dominates and the influence of the prior/regularizer $\to 0$. Makes sense: with tons of data you trust the data and stop leaning on your prior belief.
- **For MAP, the noise scale $\sigma$ matters.** In pure MLE the constant $\sigma$ dropped out of the $\arg\min$. In MAP it does **not** cancel: it sets the *relative* weight of the loss vs. the regularizer (effectively rescaling $\lambda$). So unlike MLE, the choice of $\sigma$ changes the MAP solution.

---

## 8. Summary — MLE & MAP on one page

**The core dictionary:**

| Probabilistic object | Optimization object | Example |
|---|---|---|
| likelihood $p(\mathcal{D}\mid w)$ | **loss** (via $-\log$) | Gaussian → squared, Laplace → absolute, sigmoid → logistic |
| prior $p(w)$ | **regularizer** (via $-\log$) | Gaussian → $L_2$, Laplace → $L_1$ |
| $\arg\max$ likelihood | **MLE** | minimize loss only |
| $\arg\max$ posterior | **MAP** | minimize loss + regularizer |

**The mantras:**

1. **MLE** = "make the data most probable" = **minimize the negative log-likelihood** = minimize a loss.
2. **The `log` trick** doesn't move the maximum (log is increasing) but turns nasty products into friendly sums.
3. **The choice of likelihood is the choice of loss.** Gaussian ⟺ squared, Laplace ⟺ absolute (robust because heavier-tailed), sigmoid ⟺ logistic.
4. **MAP** = "make the parameter most probable given the data" = MLE **+ a prior**.
5. **The choice of prior is the choice of regularizer**, because a regularizer is just $-\log(\text{prior})$. Gaussian prior ⟺ $L_2$, Laplace prior ⟺ $L_1$. (Laplace smoothing in naïve Bayes is regularization for the same reason.)
6. **More data ⇒ weaker prior.** The loss scales with $n$; the regularizer does not.

**Why this framework is powerful:** it doesn't just re-derive things we knew — it gives a *recipe*. Face a weird target ("number of likes on a post", "number of stars in a review")? Pick a plausible likelihood for that kind of data (e.g. a Poisson for counts), take its negative log, add the negative log of a prior on $w$, and you have a principled loss + regularizer to minimize. That is the general framework for turning complex ML tasks into optimization problems.

---

### Figure index

| Figure | File | Shows |
|---|---|---|
| 1 | `figures/fig1_coin_likelihood.png` | coin likelihood $w^2(1-w)$, peak at $\hat w = 2/3$ |
| 2 | `figures/fig2_log_monotonic.png` | log keeps the peak; NLL is minimized at the same $\hat w$ |
| 3 | `figures/fig3_gaussian_errors.png` | line fit + residual histogram ≈ Gaussian (the squared-loss assumption) |
| 4 | `figures/fig4_loss_from_likelihood.png` | Gaussian vs. Laplace densities and their $-\log$ = squared vs. absolute loss |
| 5 | `figures/fig5_sigmoid_logistic.png` | sigmoid squashing, and logistic loss vs. the 0-1 loss |
| 6 | `figures/fig6_map_posterior.png` | posterior = likelihood × prior; the prior pulls $\hat w_{\text{MAP}}$ toward 0 |
| 7 | `figures/fig7_prior_as_regularizer.png` | Gaussian prior → $L_2$ penalty, Laplace prior → $L_1$ penalty |
