# MLE & MAP — tutorial bundle

**Contents**

```
mle_map/
├── mle_map_tutorial.md               <- the tutorial (open this)
├── README.md                         <- this file
└── figures/
    ├── _style.py                     <- shared plotting style (imported by the scripts)
    ├── fig1_coin_likelihood.py       + fig1_coin_likelihood.png
    ├── fig2_log_monotonic.py         + fig2_log_monotonic.png
    ├── fig3_gaussian_errors.py       + fig3_gaussian_errors.png
    ├── fig4_loss_from_likelihood.py  + fig4_loss_from_likelihood.png
    ├── fig5_sigmoid_logistic.py      + fig5_sigmoid_logistic.png
    ├── fig6_map_posterior.py         + fig6_map_posterior.png
    └── fig7_prior_as_regularizer.py  + fig7_prior_as_regularizer.png
```

**How to read it.** Open `mle_map_tutorial.md` in any Markdown viewer that renders
LaTeX math (VS Code + "Markdown+Math", Obsidian, Typora, JupyterLab, or a
MathJax/KaTeX-enabled viewer). The images are **not** embedded in the Markdown;
each figure is shipped as a `.png` in `figures/` and referenced by name.

**Regenerating a figure.** Every picture is produced by an accurate Python script
(needs `numpy`, `scipy`, `matplotlib`):

```bash
cd figures
python3 fig6_map_posterior.py   # writes fig6_map_posterior.png
```

**Topics covered:** the likelihood function, Maximum Likelihood Estimation (MLE),
the negative log-likelihood trick, loss minimization as MLE
(squared⟺Gaussian, absolute⟺Laplace, logistic⟺sigmoid), and Maximum
A-Posteriori (MAP) estimation with priors as regularizers (Gaussian⟺L2, Laplace⟺L1).
