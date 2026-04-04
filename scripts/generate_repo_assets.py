"""Generate docs, wrappers, and notebooks for the repository."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/AKMessi/math-for-ML"

TOPICS = [
    ("01_linear_algebra", "Linear Algebra", "linear_algebra", [
        "01_vectors_and_spaces.ipynb",
        "02_matrices_and_operations.ipynb",
        "03_matrix_decompositions.ipynb",
        "04_eigenvalues_eigenvectors.ipynb",
        "05_SVD_complete.ipynb",
        "06_PCA_from_scratch.ipynb",
        "07_norms_and_distances.ipynb",
    ]),
    ("02_calculus", "Calculus", "autodiff", [
        "01_limits_and_continuity.ipynb",
        "02_derivatives_single_variable.ipynb",
        "03_partial_derivatives.ipynb",
        "04_gradients_and_directional_derivatives.ipynb",
        "05_jacobians.ipynb",
        "06_hessians.ipynb",
        "07_chain_rule_scalar_to_matrix.ipynb",
        "08_matrix_calculus.ipynb",
        "09_automatic_differentiation.ipynb",
    ]),
    ("03_probability_and_statistics", "Probability and Statistics", "distributions", [
        "01_probability_foundations.ipynb",
        "02_random_variables.ipynb",
        "03_distributions_discrete.ipynb",
        "04_distributions_continuous.ipynb",
        "05_joint_marginal_conditional.ipynb",
        "06_expectation_variance_covariance.ipynb",
        "07_MLE_and_MAP.ipynb",
        "08_bayesian_inference.ipynb",
        "09_information_theory.ipynb",
        "10_hypothesis_testing.ipynb",
    ]),
    ("04_optimization", "Optimization", "optimizers", [
        "01_convexity.ipynb",
        "02_gradient_descent.ipynb",
        "03_sgd_and_variants.ipynb",
        "04_second_order_methods.ipynb",
        "05_constrained_optimization.ipynb",
        "06_linear_programming.ipynb",
        "07_loss_landscapes.ipynb",
        "08_learning_rate_theory.ipynb",
    ]),
    ("05_numerical_methods", "Numerical Methods", "numerical", [
        "01_floating_point_arithmetic.ipynb",
        "02_numerical_stability.ipynb",
        "03_numerical_linear_algebra.ipynb",
        "04_numerical_integration.ipynb",
    ]),
    ("06_information_theory", "Information Theory", "information", [
        "01_entropy_and_information.ipynb",
        "02_KL_divergence.ipynb",
        "03_mutual_information.ipynb",
        "04_rate_distortion.ipynb",
    ]),
    ("07_graph_theory", "Graph Theory", "graphs", [
        "01_graph_basics.ipynb",
        "02_spectral_graph_theory.ipynb",
        "03_computation_graphs.ipynb",
    ]),
    ("08_ml_math_applied", "ML Math Applied", "ml_components", [
        "01_backpropagation_derived.ipynb",
        "02_attention_mechanism.ipynb",
        "03_convolutions_as_matmul.ipynb",
        "04_batch_norm_math.ipynb",
        "05_dropout_math.ipynb",
        "06_loss_functions_derived.ipynb",
        "07_regularization_math.ipynb",
        "08_transformer_math_complete.ipynb",
        "09_diffusion_models_math.ipynb",
        "10_VAE_math.ipynb",
    ]),
    ("09_fourier_and_signal", "Fourier and Signal", None, [
        "01_fourier_series.ipynb",
        "02_fourier_transform.ipynb",
        "03_convolution_theorem.ipynb",
        "04_positional_encodings.ipynb",
    ]),
]

IMPORTS = {
    "linear_algebra": "math_for_ml.linear_algebra",
    "autodiff": "math_for_ml.autodiff",
    "distributions": "math_for_ml.distributions",
    "optimizers": "math_for_ml.optimizers",
    "numerical": "math_for_ml.numerical",
    "information": "math_for_ml.information",
    "graphs": "math_for_ml.graphs",
    "ml_components": "math_for_ml.ml_components",
}

CHEATSHEETS = {
    "linear_algebra.md": "# Linear Algebra Cheatsheet\n\n- Dot product: $u^\\top v$\n- Projection: $\\frac{u^\\top v}{v^\\top v}v$\n- SVD: $A = U\\Sigma V^\\top$\n- PCA: principal directions of centered data\n",
    "matrix_calculus.md": "# Matrix Calculus Cheatsheet\n\n- Use differentials first\n- $d(a^\\top x)=a^\\top dx$\n- $\\nabla_x x^\\top A x = (A + A^\\top)x$\n- Backpropagation is repeated chain rule\n",
    "probability.md": "# Probability Cheatsheet\n\n- Expectation: $\\mathbb{E}[X]$\n- Variance: $\\mathbb{V}[X]$\n- Bayes: $p(\\theta\\mid x) \\propto p(x\\mid\\theta)p(\\theta)$\n- Entropy: $H(X)=-\\sum p\\log p$\n",
    "optimization.md": "# Optimization Cheatsheet\n\n- Gradient descent: $x_{t+1}=x_t-\\eta\\nabla f(x_t)$\n- Momentum adds velocity\n- RMSProp and Adam rescale gradients\n- Newton uses the Hessian\n",
    "common_derivatives.md": "# Common Derivatives\n\n- $(x^n)' = nx^{n-1}$\n- $(e^x)' = e^x$\n- $(\\log x)' = 1/x$\n- $\\sigma'(x)=\\sigma(x)(1-\\sigma(x))$\n- $(\\tanh x)' = 1 - \\tanh^2 x$\n",
    "greek_letters_and_notation.md": "# Greek Letters and Notation\n\n- $\\theta$: parameters\n- $\\lambda$: regularization or eigenvalue\n- $\\mu$: mean\n- $\\sigma$: standard deviation\n- $\\nabla$: gradient\n",
}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def title_from_name(filename: str) -> str:
    stem = filename.removesuffix(".ipynb").split("_", 1)[1]
    words = stem.split("_")
    upper = {"svd": "SVD", "pca": "PCA", "mle": "MLE", "map": "MAP", "kl": "KL", "vae": "VAE", "ml": "ML"}
    return " ".join(upper.get(word.lower(), word.capitalize()) for word in words)


def lesson_profile(filename: str) -> tuple[str, str, str]:
    stem = filename.lower()
    if "attention" in stem or "transformer" in stem:
        return (
            "Attention turns similarity scores into context-aware weighted averages.",
            "Like reading a sentence while deciding which earlier words deserve focus.",
            "Transformers depend on this scoring and weighting pattern.",
        )
    if "pca" in stem or "svd" in stem or "eigen" in stem:
        return (
            "This topic finds the dominant directions inside a linear transformation or dataset.",
            "Like rotating a point cloud until its structure is easiest to see.",
            "Dimensionality reduction and compression depend on these directions.",
        )
    if "autodiff" in stem or "backpropagation" in stem or "chain_rule" in stem:
        return (
            "This topic tracks how local changes combine into global gradients.",
            "Like sending influence backward through a wiring diagram.",
            "PyTorch autograd and neural network training rely on exactly this logic.",
        )
    if "probability" in stem or "bayesian" in stem or "mle" in stem or "distribution" in stem:
        return (
            "This topic turns uncertainty into a precise mathematical object.",
            "Like updating a disciplined betting table after seeing new evidence.",
            "Likelihoods, calibration, and generative models all depend on it.",
        )
    if "optimization" in stem or "gradient_descent" in stem or "learning_rate" in stem:
        return (
            "This topic explains how gradients become parameter updates.",
            "Like walking downhill with only local slope information.",
            "Training loops live or die by these update rules.",
        )
    if "graph" in stem:
        return (
            "Graphs store entities and relationships in a form algorithms can traverse.",
            "Like cities connected by roads or modules linked by dependencies.",
            "Computation graphs and GNNs use this structure directly.",
        )
    if "fourier" in stem or "convolution" in stem or "positional" in stem:
        return (
            "This topic rewrites signals in a basis where patterns become easier to analyze.",
            "Like hearing a chord as individual notes.",
            "Convolution, spectra, and positional encodings all inherit this view.",
        )
    return (
        "This lesson builds intuition before formal derivation and implementation.",
        "Treat the concept as a reusable tool, not a formula to memorize.",
        "The same idea appears inside real ML systems and training code.",
    )


def code_for_topic(topic_dir: str, module_name: str | None) -> tuple[str, str, str]:
    if topic_dir == "01_linear_algebra":
        return (
            "from src.linear_algebra import pca_from_scratch\n\nX = np.array([[2., 1.], [3., 2.], [4., 3.], [5., 3.5]])\nresult = pca_from_scratch(X, n_components=1)\nnumpy_result = result.scores\nprint(result.components)\nprint(result.explained_variance_ratio)",
            "X_t = torch.tensor(X, dtype=torch.float64)\ncentered = X_t - X_t.mean(dim=0)\n_, _, Vh = torch.linalg.svd(centered, full_matrices=False)\ntorch_result = centered @ Vh[:1].T\nprint(torch.allclose(torch_result.abs(), torch.tensor(numpy_result).abs(), atol=1e-5))",
            "plt.scatter(X[:, 0], X[:, 1])\nplt.quiver(result.mean[0], result.mean[1], result.components[0, 0], result.components[0, 1], angles='xy', scale_units='xy', scale=1, color='tab:red')\nplt.title('Principal direction')\nplt.show()",
        )
    if topic_dir == "02_calculus":
        return (
            "from src.autodiff import Value, gradient_check\n\nx = Value(1.5)\ny = (x * x + 3.0 * x + 1.0).tanh()\ny.backward()\nnumpy_result = np.array([x.grad])\nprint('autodiff gradient =', x.grad)\nprint(gradient_check(lambda nodes: (nodes[0] * nodes[0] + 3.0 * nodes[0] + 1.0).tanh(), [1.5]))",
            "x_t = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)\ny_t = torch.tanh(x_t * x_t + 3.0 * x_t + 1.0)\ny_t.backward()\ntorch_result = torch.tensor([x_t.grad], dtype=torch.float64)\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
            "grid = np.linspace(-3, 3, 200)\ncurve = np.tanh(grid**2 + 3.0 * grid + 1.0)\nplt.plot(grid, curve)\nplt.title('A differentiable objective')\nplt.show()",
        )
    if topic_dir == "03_probability_and_statistics":
        return (
            "from src.distributions import gaussian_pdf, maximum_likelihood_gaussian\n\nsamples = np.array([1.2, 0.9, 1.4, 1.1, 1.0])\nmle_mean, mle_std = maximum_likelihood_gaussian(samples)\nx = np.linspace(-1, 3, 200)\nnumpy_result = gaussian_pdf(x, mle_mean, mle_std)\nprint(mle_mean, mle_std)",
            "dist = torch.distributions.Normal(torch.tensor(mle_mean), torch.tensor(mle_std))\ntorch_result = torch.exp(dist.log_prob(torch.tensor(x, dtype=torch.float64)))\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
            "plt.hist(samples, bins=5, density=True, alpha=0.4)\nplt.plot(x, numpy_result)\nplt.title('Estimated Gaussian density')\nplt.show()",
        )
    if topic_dir == "04_optimization":
        return (
            "from src.optimizers import Adam, minimize\n\nobjective = lambda w: float(np.sum((w - 3.0) ** 2))\ngradient = lambda w: 2.0 * (w - 3.0)\nresult = minimize(objective, gradient, np.array([8.0]), Adam(learning_rate=0.2), iterations=25)\nnumpy_result = result.parameters\nprint(numpy_result)",
            "parameter = torch.tensor([8.0], dtype=torch.float64, requires_grad=True)\noptimizer = torch.optim.Adam([parameter], lr=0.2)\nfor _ in range(25):\n    optimizer.zero_grad()\n    loss = torch.sum((parameter - 3.0) ** 2)\n    loss.backward()\n    optimizer.step()\ntorch_result = parameter.detach()\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=5e-2))",
            "plt.plot(result.losses)\nplt.title('Optimization trajectory')\nplt.show()",
        )
    if topic_dir == "05_numerical_methods":
        return (
            "from src.numerical import logsumexp, safe_softmax\n\nlogits = np.array([[1000.0, 1001.0, 1002.0]])\nnumpy_result = safe_softmax(logits, axis=1)\nprint(logsumexp(logits, axis=1))\nprint(numpy_result)",
            "logits_t = torch.tensor(logits, dtype=torch.float64)\ntorch_result = torch.softmax(logits_t, dim=1)\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
            "plt.bar(np.arange(logits.shape[1]), numpy_result.ravel())\nplt.title('Stable softmax probabilities')\nplt.show()",
        )
    if topic_dir == "06_information_theory":
        return (
            "from src.information import entropy, kl_divergence\n\np = np.array([0.7, 0.3])\nq = np.array([0.4, 0.6])\nnumpy_result = np.array([entropy(p), kl_divergence(p, q)])\nprint(numpy_result)",
            "p_t = torch.tensor(p, dtype=torch.float64)\nq_t = torch.tensor(q, dtype=torch.float64)\ntorch_result = torch.tensor([-torch.sum(p_t * torch.log2(p_t)), torch.sum(p_t * torch.log2(p_t / q_t))], dtype=torch.float64)\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
            "plt.bar(['entropy', 'KL'], numpy_result)\nplt.title('Information quantities')\nplt.show()",
        )
    if topic_dir == "07_graph_theory":
        return (
            "from src.graphs import Graph\n\ngraph = Graph()\ngraph.add_edge('A', 'B', weight=1.0)\ngraph.add_edge('B', 'C', weight=2.0)\ngraph.add_edge('A', 'C', weight=4.0)\nadjacency = graph.adjacency_matrix()\nnumpy_result = adjacency\nprint(graph.bfs('A'))\nprint(graph.shortest_path('A', 'C'))",
            "torch_result = torch.tensor(adjacency, dtype=torch.float64)\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
            "plt.imshow(adjacency, cmap='Greens')\nplt.colorbar()\nplt.title('Adjacency matrix')\nplt.show()",
        )
    if topic_dir == "08_ml_math_applied":
        return (
            "from src.ml_components import scaled_dot_product_attention\n\nQ = np.array([[[1.0, 0.0], [0.0, 1.0]]])\nK = np.array([[[1.0, 0.0], [0.0, 1.0]]])\nV = np.array([[[2.0, 1.0], [0.5, 3.0]]])\noutput, weights = scaled_dot_product_attention(Q, K, V)\nnumpy_result = output\nprint(weights)",
            "Q_t = torch.tensor(Q, dtype=torch.float64)\nK_t = torch.tensor(K, dtype=torch.float64)\nV_t = torch.tensor(V, dtype=torch.float64)\nscores = Q_t @ K_t.transpose(-1, -2) / np.sqrt(Q.shape[-1])\ntorch_result = torch.softmax(scores, dim=-1) @ V_t\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
            "plt.imshow(weights[0], cmap='Oranges')\nplt.colorbar()\nplt.title('Attention weights')\nplt.show()",
        )
    return (
        "t = np.linspace(0.0, 1.0, 256, endpoint=False)\nsignal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)\nnumpy_result = np.fft.rfft(signal)\nfreqs = np.fft.rfftfreq(len(signal), d=t[1] - t[0])\nprint(freqs[np.argsort(np.abs(numpy_result))[-3:]])",
        "signal_t = torch.tensor(signal, dtype=torch.float64)\ntorch_result = torch.fft.rfft(signal_t)\nprint(torch.allclose(torch_result, torch.tensor(numpy_result), atol=1e-5))",
        "plt.plot(freqs, np.abs(numpy_result))\nplt.title('Frequency magnitude spectrum')\nplt.show()",
    )


def notebook(topic_title: str, topic_dir: str, module_name: str | None, filename: str) -> dict[str, object]:
    title = title_from_name(filename)
    intuition, analogy, ml_use = lesson_profile(filename)
    numpy_code, torch_code, viz_code = code_for_topic(topic_dir, module_name)
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": f"# {title}\n\nPart of **{topic_title}**."},
        {"cell_type": "markdown", "metadata": {}, "source": f"## 1. Intuition First\n\n{intuition}\n\n**Analogy:** {analogy}\n\n**Why this matters in ML:** {ml_use}"},
        {"cell_type": "markdown", "metadata": {}, "source": "## 2. Mathematical Foundation\n\nState the object precisely, define every symbol, and derive the working identity line by line instead of jumping to the final formula.\n\n- Write the definition first.\n- Show the intermediate algebra.\n- Explain the assumptions.\n- Flag the common misconception before moving on."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "import numpy as np\nimport torch\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-v0_8-whitegrid')\nnp.set_printoptions(precision=5, suppress=True)\ntorch.set_printoptions(precision=5)"},
        {"cell_type": "markdown", "metadata": {}, "source": "## 3. NumPy Implementation From Scratch\n\nUse the local teaching module or plain NumPy to implement the core idea explicitly."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": numpy_code},
        {"cell_type": "markdown", "metadata": {}, "source": "## 4. PyTorch Verification\n\nRebuild the same quantity in PyTorch and verify equivalence with `torch.allclose`."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": torch_code},
        {"cell_type": "markdown", "metadata": {}, "source": "## 5. Visualization\n\nConnect the formula to geometry, distributions, optimization behavior, or spectral structure."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": viz_code},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "try:\n    import ipywidgets as widgets\n    widgets.IntSlider(description='Explore', min=0, max=10, value=5)\nexcept Exception as exc:\n    print('ipywidgets is optional for this notebook:', exc)"},
        {"cell_type": "markdown", "metadata": {}, "source": "## 6. Where This Lives in Real Models\n\n```python\noutput = module(inputs)\nloss = criterion(output, targets)\nloss.backward()\n```\n\nIf the math is wrong here, the implementation drifts even when the code still runs."},
        {"cell_type": "markdown", "metadata": {}, "source": f"## 7. Exercises\n\n1. Beginner: compute a tiny hand-worked example for **{title}**.\n2. Intermediate: modify the demo inputs and predict the change before running the cell.\n3. Advanced: rebuild the same computation without using the helper function.\n\n<details>\n<summary>Hidden solution ideas</summary>\n\n- Start from the definition.\n- Check dimensions and edge cases.\n- Verify the result again in PyTorch.\n</details>"},
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def topic_readme(title: str, module_name: str | None, notebooks: list[str]) -> str:
    lines = [f"# {title}", "", "## Lessons", ""]
    lines += [f"- [{title_from_name(nb)}](./{nb})" for nb in notebooks]
    if module_name:
        lines += ["", "## Local Module", "", f"`src/{module_name}.py` re-exports `{IMPORTS[module_name]}` for notebook-local imports."]
    lines += ["", "## Suggested Order", "", "Work through the numbered notebooks in sequence."]
    return "\n".join(lines)


def root_readme() -> str:
    topic_rows = "\n".join(
        f"| [{title}](./{directory}/README.md) | {len(notebooks)} notebooks |"
        for directory, title, _, notebooks in TOPICS
    )
    return f"""# Mathematics for Machine Learning

[![build](https://img.shields.io/badge/build-pytest%20ready-0A7B83)](./README.md)
[![license](https://img.shields.io/badge/license-MIT-1F6FEB)](./LICENSE)
[![contributions](https://img.shields.io/badge/contributions-welcome-2DA44E)](./CONTRIBUTING.md)

A beginner-to-advanced open-source curriculum for the mathematics that underlies modern machine learning.

Repository: {REPOSITORY_URL}

This repository is designed for:
- Complete beginners who need intuition before notation
- Practitioners who want to replace memorized formulas with derivations
- Advanced engineers who want tested reference implementations
- Researchers who need a compact, reusable math companion

## Learning Path

```mermaid
flowchart TD
    A[00 How To Use] --> B[01 Linear Algebra]
    B --> C[02 Calculus]
    C --> D[03 Probability and Statistics]
    B --> E[04 Optimization]
    B --> F[05 Numerical Methods]
    D --> G[06 Information Theory]
    B --> H[07 Graph Theory]
    B --> I[08 ML Math Applied]
    B --> J[09 Fourier and Signal]
```

## Tracks

| Track | Audience | Time |
| --- | --- | --- |
| Beginner | No ML background | 8 weeks |
| Practitioner | Knows ML, wants rigor | 4 weeks |
| Researcher | Needs deeper theory | 6 weeks |
| Quick Reference | Needs formulas fast | Instant |

## Prerequisites

| Need | Start here |
| --- | --- |
| Algebra | [prerequisites.md](./00_how_to_use/prerequisites.md) |
| Python basics | [prerequisites.md](./00_how_to_use/prerequisites.md) |
| Study plan | [learning_paths.md](./00_how_to_use/learning_paths.md) |

## One-Command Setup

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
```

Run the test suite with:

```bash
python -m pytest
```

## Sections

| Section | Scope |
| --- | --- |
{topic_rows}

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Citation

```bibtex
@misc{{math_for_ml_repo,
  title = {{Mathematics for Machine Learning}},
  author = {{Open Source Contributors}},
  year = {{2026}},
  howpublished = {{GitHub repository}},
  url = {{{REPOSITORY_URL}}}
}}
```

## License

MIT. See [LICENSE](./LICENSE).
"""


def generate() -> None:
    write_text(ROOT / "README.md", root_readme())
    write_text(
        ROOT / "CONTRIBUTING.md",
        "# Contributing\n\n## Standards\n\n- Put intuition before equations.\n- Derive results instead of skipping steps.\n- Keep implementations readable, typed, and documented.\n- Verify important calculations against PyTorch.\n- Add or update pytest coverage for source changes.\n\n## Workflow\n\n1. Install dependencies with `python -m pip install -r requirements.txt`.\n2. Install the package with `python -m pip install -e .`.\n3. Run `python -m pytest`.\n4. If you changed notebooks, run them top to bottom before opening a pull request.\n",
    )
    write_text(
        ROOT / "ROADMAP.md",
        "# Roadmap\n\n## Done\n\n- [x] Installable `math_for_ml` package\n- [x] Topic scaffolding and 59 notebooks\n- [x] Notebook-local wrapper modules\n- [x] Pytest coverage for the main source modules\n- [x] Cheatsheets and learner guidance\n\n## Next\n\n- [ ] Expand more notebooks with longer derivations and custom figures\n- [ ] Add notebook execution checks in CI\n- [ ] Add more reusable diagrams under `assets/diagrams/`\n- [ ] Add richer model-specific case studies in the applied section\n",
    )
    write_text(ROOT / "LICENSE", "MIT License\n\nCopyright (c) 2026 Open Source Contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.")
    write_text(ROOT / "00_how_to_use" / "learning_paths.md", "# Learning Paths\n\n- Beginner: start with linear algebra, calculus, probability, then applied ML math.\n- Practitioner: jump to SVD, matrix calculus, optimization, numerical stability, and applied notebooks.\n- Researcher: read section README files, then prioritize Jacobians, Hessians, Bayesian inference, information theory, spectral graphs, transformers, diffusion, and VAEs.\n- Quick Reference: use `cheatsheets/` and topic README files.\n")
    write_text(ROOT / "00_how_to_use" / "prerequisites.md", "# Prerequisites\n\n- Basic algebra and equation manipulation\n- Basic Python and package installation\n- Willingness to work through code, not just formulas\n\nIf something feels abstract, move to the implementation or visualization section first, then return to the derivation.\n")
    for name, content in CHEATSHEETS.items():
        write_text(ROOT / "cheatsheets" / name, content)
    write_text(ROOT / "assets" / "diagrams" / "README.md", "# Diagrams\n\nStore reusable visuals for notebooks here.\n")

    for directory, title, module_name, notebooks in TOPICS:
        topic_path = ROOT / directory
        write_text(topic_path / "README.md", topic_readme(title, module_name, notebooks))
        if module_name:
            write_text(topic_path / "src" / "__init__.py", '"""Notebook-local helper package."""')
            write_text(
                topic_path / "src" / f"{module_name}.py",
                f'"""Notebook-facing re-export of `{IMPORTS[module_name]}`."""\n\nfrom {IMPORTS[module_name]} import *  # noqa: F401,F403\nfrom {IMPORTS[module_name]} import __all__\n',
            )
        for filename in notebooks:
            write_notebook(topic_path / filename, notebook(title, directory, module_name, filename))


if __name__ == "__main__":
    generate()
