# Regression Variants — tutorial bundle

**Contents**

```
regression_variants/
├── regression_variants_tutorial.md   <- the tutorial (open this)
├── README.md                         <- this file
└── figures/
    ├── _style.py                     <- shared plotting style (imported by the scripts)
    ├── fig1_norm_penalties.py        + fig1_norm_penalties.png
    ├── fig2_sparsity_geometry.py     + fig2_sparsity_geometry.png
    ├── fig3_regularization_path.py   + fig3_regularization_path.png
    ├── fig4_robust_losses.py         + fig4_robust_losses.png
    ├── fig5_outlier_fit.py           + fig5_outlier_fit.png
    └── fig6_robustness_zoo.py        + fig6_robustness_zoo.png
```

**How to read it.** Open `regression_variants_tutorial.md` in any Markdown viewer
that renders LaTeX math (VS Code + "Markdown+Math", Obsidian, Typora, JupyterLab,
or a MathJax/KaTeX-enabled viewer). The images are **not** embedded in the
Markdown; each figure is shipped as a `.png` in `figures/` and referenced by name.

**Regenerating a figure.** Every picture is produced by an accurate Python script
(needs `numpy`, `scipy`, `scikit-learn`, `matplotlib`):

```bash
cd figures
python3 fig3_regularization_path.py   # writes fig3_regularization_path.png
```

**Topics covered:** L1/LASSO regularization and why it creates sparsity (vs. L2),
the regularization path, robust regression (absolute error, Huber loss),
and the very-robust (square-root) and brittle (L∞) extremes.
