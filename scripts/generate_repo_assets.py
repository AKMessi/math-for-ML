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

DIAGRAMS = {
    "attention_pipeline.svg": """
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">
      <rect width="900" height="260" fill="#f8fbfd"/>
      <text x="40" y="44" fill="#133c55" font-size="26" font-family="Arial">Scaled Dot-Product Attention</text>
      <rect x="40" y="90" width="130" height="80" rx="16" fill="#dceef8" stroke="#3c6e71" stroke-width="2"/>
      <rect x="210" y="90" width="130" height="80" rx="16" fill="#dceef8" stroke="#3c6e71" stroke-width="2"/>
      <rect x="380" y="90" width="150" height="80" rx="16" fill="#f4e8c1" stroke="#c98f2b" stroke-width="2"/>
      <rect x="590" y="90" width="130" height="80" rx="16" fill="#e1f2df" stroke="#4f772d" stroke-width="2"/>
      <rect x="760" y="90" width="100" height="80" rx="16" fill="#ffe2cc" stroke="#c16630" stroke-width="2"/>
      <text x="92" y="138" fill="#133c55" font-size="24" font-family="Arial">Q</text>
      <text x="262" y="138" fill="#133c55" font-size="24" font-family="Arial">K</text>
      <text x="412" y="124" fill="#7c4f00" font-size="20" font-family="Arial">scores = QK^T / sqrt(d_k)</text>
      <text x="628" y="124" fill="#335c34" font-size="20" font-family="Arial">softmax</text>
      <text x="776" y="124" fill="#8f3d13" font-size="20" font-family="Arial">V</text>
      <text x="770" y="150" fill="#8f3d13" font-size="20" font-family="Arial">context</text>
      <path d="M170 130 H210" stroke="#133c55" stroke-width="4" fill="none"/>
      <path d="M340 130 H380" stroke="#133c55" stroke-width="4" fill="none"/>
      <path d="M530 130 H590" stroke="#7c4f00" stroke-width="4" fill="none"/>
      <path d="M720 130 H760" stroke="#335c34" stroke-width="4" fill="none"/>
      <path d="M810 170 V205 H470 V170" stroke="#8f3d13" stroke-width="3" fill="none" stroke-dasharray="8 6"/>
      <text x="455" y="225" fill="#8f3d13" font-size="16" font-family="Arial">weights multiply values</text>
    </svg>
    """,
    "backprop_chain_rule.svg": """
    <svg xmlns="http://www.w3.org/2000/svg" width="940" height="260" viewBox="0 0 940 260">
      <rect width="940" height="260" fill="#faf9f5"/>
      <text x="40" y="42" fill="#23395b" font-size="26" font-family="Arial">Backpropagation Through a Tiny MLP</text>
      <rect x="40" y="90" width="130" height="82" rx="16" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
      <rect x="220" y="90" width="150" height="82" rx="16" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
      <rect x="430" y="90" width="130" height="82" rx="16" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
      <rect x="620" y="90" width="130" height="82" rx="16" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
      <rect x="800" y="90" width="100" height="82" rx="16" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
      <text x="94" y="138" font-size="20" font-family="Arial" fill="#1d4ed8">X</text>
      <text x="258" y="124" font-size="18" font-family="Arial" fill="#0c4a6e">z1 = XW1 + b1</text>
      <text x="470" y="138" font-size="18" font-family="Arial" fill="#166534">h = ReLU(z1)</text>
      <text x="652" y="124" font-size="18" font-family="Arial" fill="#9a3412">y_hat = hW2 + b2</text>
      <text x="825" y="138" font-size="20" font-family="Arial" fill="#991b1b">L</text>
      <path d="M170 131 H220" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M370 131 H430" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M560 131 H620" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M750 131 H800" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M850 185 H700 H495 H285 H105" stroke="#dc2626" stroke-width="4" fill="none" stroke-dasharray="10 7"/>
      <text x="612" y="204" font-size="16" font-family="Arial" fill="#991b1b">gradients travel backward by the chain rule</text>
    </svg>
    """,
    "diffusion_forward_reverse.svg": """
    <svg xmlns="http://www.w3.org/2000/svg" width="980" height="260" viewBox="0 0 980 260">
      <rect width="980" height="260" fill="#fbfaf8"/>
      <text x="40" y="42" fill="#264653" font-size="26" font-family="Arial">Diffusion: Forward Noise and Reverse Denoising</text>
      <circle cx="110" cy="132" r="44" fill="#cdeef3" stroke="#2a9d8f" stroke-width="2"/>
      <circle cx="300" cy="132" r="44" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
      <circle cx="490" cy="132" r="44" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>
      <circle cx="680" cy="132" r="44" fill="#fde68a" stroke="#d97706" stroke-width="2"/>
      <circle cx="870" cy="132" r="44" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
      <text x="84" y="138" font-size="20" font-family="Arial" fill="#0f766e">x0</text>
      <text x="274" y="138" font-size="20" font-family="Arial" fill="#1d4ed8">x_t</text>
      <text x="448" y="138" font-size="18" font-family="Arial" fill="#6d28d9">eps_theta</text>
      <text x="646" y="138" font-size="18" font-family="Arial" fill="#9a3412">x_hat_0</text>
      <text x="842" y="138" font-size="18" font-family="Arial" fill="#991b1b">x_t-1</text>
      <path d="M154 132 H256" stroke="#264653" stroke-width="4" fill="none"/>
      <path d="M344 132 H446" stroke="#264653" stroke-width="4" fill="none"/>
      <path d="M534 132 H636" stroke="#264653" stroke-width="4" fill="none"/>
      <path d="M724 132 H826" stroke="#264653" stroke-width="4" fill="none"/>
      <text x="173" y="114" font-size="16" font-family="Arial" fill="#264653">add Gaussian noise</text>
      <text x="350" y="114" font-size="16" font-family="Arial" fill="#264653">predict epsilon</text>
      <text x="548" y="114" font-size="16" font-family="Arial" fill="#264653">recover clean signal</text>
      <text x="742" y="114" font-size="16" font-family="Arial" fill="#264653">reverse step</text>
    </svg>
    """,
    "pca_projection.svg": """
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">
      <rect width="900" height="260" fill="#fbfcfe"/>
      <text x="40" y="42" fill="#1d3557" font-size="26" font-family="Arial">PCA as Projection Onto the Dominant Direction</text>
      <line x1="110" y1="185" x2="320" y2="70" stroke="#457b9d" stroke-width="6"/>
      <line x1="180" y1="210" x2="260" y2="105" stroke="#adb5bd" stroke-width="3" stroke-dasharray="8 6"/>
      <circle cx="180" cy="210" r="8" fill="#e63946"/>
      <circle cx="242" cy="129" r="8" fill="#2a9d8f"/>
      <text x="332" y="76" font-size="18" font-family="Arial" fill="#1d3557">principal axis</text>
      <text x="170" y="230" font-size="16" font-family="Arial" fill="#e63946">original point</text>
      <text x="220" y="120" font-size="16" font-family="Arial" fill="#2a9d8f">projection</text>
      <circle cx="500" cy="155" r="8" fill="#457b9d"/>
      <circle cx="540" cy="135" r="8" fill="#457b9d"/>
      <circle cx="575" cy="118" r="8" fill="#457b9d"/>
      <circle cx="610" cy="102" r="8" fill="#457b9d"/>
      <circle cx="650" cy="86" r="8" fill="#457b9d"/>
      <circle cx="520" cy="175" r="8" fill="#457b9d"/>
      <circle cx="560" cy="155" r="8" fill="#457b9d"/>
      <circle cx="595" cy="140" r="8" fill="#457b9d"/>
      <circle cx="630" cy="121" r="8" fill="#457b9d"/>
      <text x="470" y="210" font-size="18" font-family="Arial" fill="#1d3557">variance concentrates along one direction</text>
    </svg>
    """,
    "transformer_block.svg": """
    <svg xmlns="http://www.w3.org/2000/svg" width="980" height="260" viewBox="0 0 980 260">
      <rect width="980" height="260" fill="#f8fafc"/>
      <text x="40" y="42" fill="#0f172a" font-size="26" font-family="Arial">Transformer Block</text>
      <rect x="60" y="92" width="130" height="82" rx="16" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
      <rect x="250" y="92" width="170" height="82" rx="16" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
      <rect x="490" y="92" width="130" height="82" rx="16" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
      <rect x="680" y="92" width="170" height="82" rx="16" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
      <rect x="880" y="92" width="60" height="82" rx="16" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
      <text x="86" y="138" font-size="20" font-family="Arial" fill="#1d4ed8">tokens</text>
      <text x="278" y="138" font-size="18" font-family="Arial" fill="#0c4a6e">masked attention</text>
      <text x="518" y="138" font-size="18" font-family="Arial" fill="#166534">add + norm</text>
      <text x="708" y="138" font-size="18" font-family="Arial" fill="#9a3412">feed-forward</text>
      <text x="891" y="138" font-size="18" font-family="Arial" fill="#991b1b">y</text>
      <path d="M190 133 H250" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M420 133 H490" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M620 133 H680" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M850 133 H880" stroke="#334155" stroke-width="4" fill="none"/>
      <path d="M125 200 V210 H555 V174" stroke="#dc2626" stroke-width="4" fill="none" stroke-dasharray="9 7"/>
      <path d="M555 200 V210 H910 V174" stroke="#dc2626" stroke-width="4" fill="none" stroke-dasharray="9 7"/>
      <text x="165" y="232" font-size="16" font-family="Arial" fill="#991b1b">residual paths preserve token information while new mixing is learned</text>
    </svg>
    """,
    "vae_elbo.svg": """
    <svg xmlns="http://www.w3.org/2000/svg" width="980" height="260" viewBox="0 0 980 260">
      <rect width="980" height="260" fill="#fffaf5"/>
      <text x="40" y="42" fill="#5f0f40" font-size="26" font-family="Arial">Variational Autoencoder ELBO</text>
      <rect x="60" y="92" width="140" height="82" rx="16" fill="#fde2e4" stroke="#d62839" stroke-width="2"/>
      <rect x="260" y="92" width="180" height="82" rx="16" fill="#f9dcc4" stroke="#bc6c25" stroke-width="2"/>
      <rect x="510" y="92" width="140" height="82" rx="16" fill="#d8f3dc" stroke="#2d6a4f" stroke-width="2"/>
      <rect x="710" y="92" width="200" height="82" rx="16" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
      <text x="117" y="138" font-size="20" font-family="Arial" fill="#9d0208">x</text>
      <text x="302" y="124" font-size="18" font-family="Arial" fill="#7f5539">encoder -> mu, log_var</text>
      <text x="554" y="138" font-size="18" font-family="Arial" fill="#1b4332">z = mu + sigma * eps</text>
      <text x="734" y="124" font-size="18" font-family="Arial" fill="#1d4ed8">decoder + ELBO</text>
      <text x="734" y="150" font-size="16" font-family="Arial" fill="#1d4ed8">reconstruction - beta * KL</text>
      <path d="M200 133 H260" stroke="#5f0f40" stroke-width="4" fill="none"/>
      <path d="M440 133 H510" stroke="#5f0f40" stroke-width="4" fill="none"/>
      <path d="M650 133 H710" stroke="#5f0f40" stroke-width="4" fill="none"/>
      <path d="M600 78 C650 40 760 40 820 78" stroke="#2d6a4f" stroke-width="3" fill="none" stroke-dasharray="8 6"/>
      <text x="655" y="56" font-size="16" font-family="Arial" fill="#2d6a4f">latent regularization</text>
    </svg>
    """,
}

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

BASE_IMPORTS = """import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5)
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()}


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip(),
    }


def notebook_document(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


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


def pca_notebook() -> dict[str, object]:
    return notebook_document([
        markdown_cell("# PCA From Scratch\n\nPart of **Linear Algebra**."),
        markdown_cell(
            """
            ## 1. Intuition First

            PCA keeps the direction that preserves as much centered variance as possible when you compress the data to a line.

            ![PCA projection](../assets/diagrams/pca_projection.svg)
            """
        ),
        code_cell(BASE_IMPORTS),
        markdown_cell(
            r"""
            ## 2. Variance Maximization Derivation

            Let the centered data matrix be $X_c \in \mathbb{R}^{n \times d}$ and let $u \in \mathbb{R}^d$ be a unit vector.
            The variance of the projected dataset is

            $$
            \mathrm{Var}(X_c u) = \frac{1}{n} u^\top X_c^\top X_c u.
            $$

            If we define $\Sigma = \frac{1}{n} X_c^\top X_c$, then PCA solves

            $$
            \max_{\|u\|_2 = 1} u^\top \Sigma u.
            $$

            Using a Lagrange multiplier $\lambda$,

            $$
            \mathcal{L}(u, \lambda) = u^\top \Sigma u - \lambda (u^\top u - 1),
            $$

            and differentiating gives

            $$
            \Sigma u = \lambda u.
            $$

            So the first principal component is the eigenvector with the largest eigenvalue.
            """
        ),
        code_cell(
            """
            from src.linear_algebra import pca_from_scratch

            angles = np.linspace(-2.5, 2.5, 14)
            X = np.column_stack((2.0 * angles + 0.25 * np.sin(angles), 1.1 * angles + 0.45 * np.cos(angles)))
            result = pca_from_scratch(X, n_components=2)
            centered = X - result.mean
            first_axis = result.components[0]
            line_coordinates = centered @ first_axis
            projected = np.outer(line_coordinates, first_axis) + result.mean

            print("principal axis =", first_axis)
            print("explained variance ratio =", result.explained_variance_ratio)
            """
        ),
        markdown_cell("## 3. PyTorch Verification\n\nThe SVD of the centered matrix gives the same principal directions as the covariance eigendecomposition."),
        code_cell(
            """
            X_t = torch.tensor(X, dtype=torch.float64)
            centered_t = X_t - X_t.mean(dim=0)
            _, singular_values, Vh = torch.linalg.svd(centered_t, full_matrices=False)
            torch_axis = Vh[0]

            print(torch.allclose(torch.tensor(first_axis).abs(), torch_axis.abs(), atol=1e-6))
            print("singular values =", singular_values)
            """
        ),
        markdown_cell("## 4. Custom Figure\n\nThe dashed connectors show the orthogonal projection of each point onto the first component."),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X[:, 0], X[:, 1], color="tab:blue", label="original points")
            ax.scatter(projected[:, 0], projected[:, 1], color="tab:orange", label="projected points")
            for original, target in zip(X, projected):
                ax.plot([original[0], target[0]], [original[1], target[1]], linestyle="--", color="#94a3b8", linewidth=1.5)
            ax.quiver(result.mean[0], result.mean[1], first_axis[0], first_axis[1], angles="xy", scale_units="xy", scale=0.3, color="tab:red", width=0.007)
            ax.set_title("PCA projects data onto a maximal-variance axis")
            ax.set_xlabel("feature 1")
            ax.set_ylabel("feature 2")
            ax.legend()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Where This Shows Up

            - data whitening before optimization
            - low-rank approximations for embeddings and activations
            - latent compression and visualization
            """
        ),
    ])


def backpropagation_notebook() -> dict[str, object]:
    return notebook_document([
        markdown_cell("# Backpropagation Derived\n\nPart of **ML Math Applied**."),
        markdown_cell(
            """
            ## 1. Intuition First

            Backpropagation is the multivariable chain rule executed in reverse topological order.

            ![Backpropagation chain rule](../assets/diagrams/backprop_chain_rule.svg)
            """
        ),
        code_cell(BASE_IMPORTS),
        markdown_cell(
            r"""
            ## 2. Derive the Gradients for a Two-Layer MLP

            Consider

            $$
            z_1 = XW_1 + b_1,
            \qquad
            h = \mathrm{ReLU}(z_1),
            \qquad
            \hat{y} = hW_2 + b_2,
            $$

            with mean-squared loss

            $$
            L = \frac{1}{2m} \|\hat{y} - y\|_2^2.
            $$

            Then

            $$
            \frac{\partial L}{\partial \hat{y}} = \frac{1}{m} (\hat{y} - y),
            \quad
            \frac{\partial L}{\partial W_2} = h^\top \frac{\partial L}{\partial \hat{y}},
            \quad
            \frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} W_2^\top.
            $$

            ReLU contributes a diagonal Jacobian:

            $$
            \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \odot \mathbf{1}[z_1 > 0],
            $$

            and the first layer gradients become

            $$
            \frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial z_1},
            \qquad
            \frac{\partial L}{\partial b_1} = \sum_i \frac{\partial L}{\partial z_1^{(i)}}.
            $$
            """
        ),
        code_cell(
            """
            X = np.array([[1.0, -1.0], [0.5, 2.0], [-1.5, 1.0]])
            y = np.array([[0.6], [-0.2], [0.8]])

            W1 = np.array([[0.4, -0.3, 0.2], [0.1, 0.5, -0.4]])
            b1 = np.array([0.05, -0.1, 0.2])
            W2 = np.array([[0.7], [-0.2], [0.5]])
            b2 = np.array([0.1])

            z1 = X @ W1 + b1
            h = np.maximum(z1, 0.0)
            y_hat = h @ W2 + b2
            residual = y_hat - y
            loss = 0.5 * np.mean(residual**2)

            dL_dyhat = residual / len(X)
            dL_dW2 = h.T @ dL_dyhat
            dL_db2 = dL_dyhat.sum(axis=0)
            dL_dh = dL_dyhat @ W2.T
            dL_dz1 = dL_dh * (z1 > 0.0)
            dL_dW1 = X.T @ dL_dz1
            dL_db1 = dL_dz1.sum(axis=0)

            print("loss =", loss)
            print("dL/dW1 =\\n", dL_dW1)
            print("dL/dW2 =\\n", dL_dW2)
            """
        ),
        markdown_cell("## 3. PyTorch Verification\n\nPyTorch should produce the same gradients when we rebuild the exact graph."),
        code_cell(
            """
            X_t = torch.tensor(X, dtype=torch.float64)
            y_t = torch.tensor(y, dtype=torch.float64)
            W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
            b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
            W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
            b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

            y_hat_t = torch.relu(X_t @ W1_t + b1_t) @ W2_t + b2_t
            loss_t = 0.5 * torch.mean((y_hat_t - y_t) ** 2)
            loss_t.backward()

            print(torch.allclose(torch.tensor(dL_dW1), W1_t.grad, atol=1e-8))
            print(torch.allclose(torch.tensor(dL_dW2), W2_t.grad, atol=1e-8))
            print(torch.allclose(torch.tensor(dL_db1), b1_t.grad, atol=1e-8))
            print(torch.allclose(torch.tensor(dL_db2), b2_t.grad, atol=1e-8))
            """
        ),
        markdown_cell("## 4. Custom Figure\n\nThe arrow labels show how gradient norms change as the chain rule propagates error backward."),
        code_cell(
            """
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.axis("off")

            nodes = [((0.08, 0.5), "X"), ((0.30, 0.5), "z1"), ((0.50, 0.5), "h"), ((0.70, 0.5), "y_hat"), ((0.90, 0.5), "L")]
            for (x, y0), label in nodes:
                box = FancyBboxPatch((x - 0.055, y0 - 0.11), 0.11, 0.22, boxstyle="round,pad=0.02", facecolor="#e0f2fe", edgecolor="#0f766e", linewidth=2)
                ax.add_patch(box)
                ax.text(x, y0, label, ha="center", va="center", fontsize=14)

            arrows = [((0.135, 0.5), (0.245, 0.5), "W1"), ((0.355, 0.5), (0.445, 0.5), "ReLU"), ((0.555, 0.5), (0.645, 0.5), "W2"), ((0.755, 0.5), (0.845, 0.5), "MSE")]
            for start, end, label in arrows:
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20, linewidth=2.5, color="#334155"))
                ax.text((start[0] + end[0]) / 2, 0.58, label, ha="center", fontsize=11)

            backward = [
                ((0.845, 0.28), (0.755, 0.28), f"||dL/dy_hat|| = {np.linalg.norm(dL_dyhat):.3f}"),
                ((0.645, 0.28), (0.555, 0.28), f"||dL/dW2|| = {np.linalg.norm(dL_dW2):.3f}"),
                ((0.445, 0.28), (0.355, 0.28), f"||dL/dz1|| = {np.linalg.norm(dL_dz1):.3f}"),
                ((0.245, 0.28), (0.135, 0.28), f"||dL/dW1|| = {np.linalg.norm(dL_dW1):.3f}"),
            ]
            for start, end, label in backward:
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color="#dc2626"))
                ax.text((start[0] + end[0]) / 2, 0.18, label, ha="center", fontsize=10, color="#991b1b")

            ax.set_title("Backpropagation is the chain rule applied in reverse")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Case Study: Gradient Flow in Deep Nets

            - if many local Jacobians are smaller than 1, gradients vanish
            - if many are larger than 1, gradients explode
            - residual connections and normalization layers help keep these products well-scaled
            """
        ),
    ])


def attention_notebook() -> dict[str, object]:
    return notebook_document([
        markdown_cell("# Attention Mechanism\n\nPart of **ML Math Applied**."),
        markdown_cell(
            """
            ## 1. Intuition First

            Attention compares each query to all keys, normalizes those comparisons, and uses the resulting weights to average the values.

            ![Attention pipeline](../assets/diagrams/attention_pipeline.svg)
            """
        ),
        code_cell(BASE_IMPORTS),
        markdown_cell(
            r"""
            ## 2. Derivation

            For query matrix $Q \in \mathbb{R}^{n_q \times d_k}$, key matrix $K \in \mathbb{R}^{n_k \times d_k}$, and value matrix $V \in \mathbb{R}^{n_k \times d_v}$:

            $$
            S = \frac{QK^\top}{\sqrt{d_k}}
            $$

            gives pairwise similarity scores. Applying a row-wise softmax turns each query row into a probability distribution:

            $$
            A = \mathrm{softmax}(S).
            $$

            The final context vectors are then

            $$
            C = AV.
            $$

            In causal language modeling, a lower-triangular mask sets all future-token scores to $-\infty$ before softmax.
            """
        ),
        code_cell(
            """
            from src.ml_components import attention_scores, causal_attention_mask, scaled_dot_product_attention

            tokens = ["the", "robot", "solves", "math"]
            Q = np.array([[[1.0, 0.2, 0.0], [0.9, 0.8, 0.1], [0.2, 1.0, 0.8], [0.1, 0.7, 1.2]]])
            K = np.array([[[1.1, 0.1, 0.1], [0.7, 0.9, 0.0], [0.3, 0.9, 0.7], [0.1, 0.8, 1.1]]])
            V = np.array([[[1.0, 0.0], [0.7, 0.4], [0.2, 1.0], [0.1, 1.3]]])

            causal_mask = causal_attention_mask(len(tokens))[None, :, :]
            raw_scores = attention_scores(Q, K)
            context, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

            print("masked weights =\\n", weights[0])
            print("last-token context =", context[0, -1])
            """
        ),
        markdown_cell("## 3. PyTorch Verification\n\nThe same masked attention computation in PyTorch should agree elementwise."),
        code_cell(
            """
            Q_t = torch.tensor(Q, dtype=torch.float64)
            K_t = torch.tensor(K, dtype=torch.float64)
            V_t = torch.tensor(V, dtype=torch.float64)
            mask_t = torch.tensor(causal_mask)

            scores_t = Q_t @ K_t.transpose(-1, -2) / np.sqrt(Q.shape[-1])
            scores_t = torch.where(mask_t, scores_t, torch.full_like(scores_t, -1e9))
            context_t = torch.softmax(scores_t, dim=-1) @ V_t

            print(torch.allclose(context_t, torch.tensor(context), atol=1e-8))
            """
        ),
        markdown_cell("## 4. Custom Figure\n\nThe heatmap shows who can attend to whom after the causal mask is applied."),
        code_cell(
            """
            entropy = -(weights[0] * np.log(np.clip(weights[0], 1e-12, None))).sum(axis=-1)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            im = axes[0].imshow(weights[0], cmap="YlOrBr", vmin=0.0, vmax=1.0)
            axes[0].set_xticks(range(len(tokens)), tokens)
            axes[0].set_yticks(range(len(tokens)), tokens)
            axes[0].set_title("Causal attention weights")
            for row in range(len(tokens)):
                for col in range(len(tokens)):
                    axes[0].text(col, row, f"{weights[0, row, col]:.2f}", ha="center", va="center", fontsize=9)

            axes[1].plot(tokens, entropy, marker="o", linewidth=2.5, color="tab:blue")
            axes[1].set_ylim(0.0, max(entropy) + 0.15)
            axes[1].set_title("Row entropy after masking")
            axes[1].set_ylabel("nats")
            axes[1].grid(True, alpha=0.3)

            fig.colorbar(im, ax=axes[0], fraction=0.046)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Case Study: Next-Token Prediction

            The last token can aggregate information from all previous positions, but the first token can only attend to itself.
            That asymmetry is exactly what autoregressive decoding needs.
            """
        ),
    ])


def transformer_notebook() -> dict[str, object]:
    return notebook_document([
        markdown_cell("# Transformer Math Complete\n\nPart of **ML Math Applied**."),
        markdown_cell(
            """
            ## 1. Intuition First

            A transformer block combines attention, residual paths, and normalization so token information can mix without becoming numerically unstable.

            ![Transformer block](../assets/diagrams/transformer_block.svg)
            """
        ),
        code_cell(BASE_IMPORTS),
        markdown_cell(
            r"""
            ## 2. Block Equations

            For token states $X$:

            $$
            \mathrm{Attention}(X) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
            $$

            where $M$ is the causal mask. A transformer block then applies

            $$
            H_1 = \mathrm{LayerNorm}(X + \mathrm{Attention}(X)),
            \qquad
            H_2 = \mathrm{LayerNorm}(H_1 + \mathrm{FFN}(H_1)).
            $$

            Residual adds preserve the incoming token state while the new sublayer learns a correction.
            """
        ),
        code_cell(
            """
            from src.ml_components import causal_attention_mask, layer_norm_forward, scaled_dot_product_attention

            def gelu_numpy(values: np.ndarray) -> np.ndarray:
                return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)))

            tokens = np.array([[[1.0, 0.3, -0.2], [0.8, 0.4, 0.1], [0.2, 1.0, 0.6], [0.0, 0.7, 1.1]]])
            W_q = np.array([[0.8, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.3, 0.9]])
            W_k = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.0, 0.2, 0.9]])
            W_v = np.array([[0.6, 0.0, 0.1], [0.2, 0.8, 0.0], [0.1, 0.1, 0.7]])
            W1 = np.array([[0.5, -0.2, 0.3, 0.1], [0.2, 0.6, -0.1, 0.4], [0.3, 0.1, 0.7, -0.2]])
            b1 = np.array([0.0, -0.1, 0.2, 0.05])
            W2 = np.array([[0.6, 0.1, -0.2], [0.2, 0.5, 0.1], [-0.1, 0.3, 0.6], [0.4, -0.2, 0.2]])
            b2 = np.array([0.05, -0.05, 0.1])

            Q = tokens @ W_q
            K = tokens @ W_k
            V = tokens @ W_v
            mask = causal_attention_mask(tokens.shape[1])[None, :, :]

            attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
            residual1 = tokens + attention_output
            norm1, _ = layer_norm_forward(residual1)
            ff_hidden = gelu_numpy(norm1 @ W1 + b1)
            ff_output = ff_hidden @ W2 + b2
            block_output, _ = layer_norm_forward(norm1 + ff_output)

            print("last token after block =", block_output[0, -1])
            print("attention for last token =", attention_weights[0, -1])
            """
        ),
        markdown_cell("## 3. PyTorch Verification\n\nRebuild the same masked-attention and feed-forward block in PyTorch."),
        code_cell(
            """
            tokens_t = torch.tensor(tokens, dtype=torch.float64)
            W_q_t = torch.tensor(W_q, dtype=torch.float64)
            W_k_t = torch.tensor(W_k, dtype=torch.float64)
            W_v_t = torch.tensor(W_v, dtype=torch.float64)
            W1_t = torch.tensor(W1, dtype=torch.float64)
            b1_t = torch.tensor(b1, dtype=torch.float64)
            W2_t = torch.tensor(W2, dtype=torch.float64)
            b2_t = torch.tensor(b2, dtype=torch.float64)
            mask_t = torch.tensor(mask)

            Q_t = tokens_t @ W_q_t
            K_t = tokens_t @ W_k_t
            V_t = tokens_t @ W_v_t
            scores_t = Q_t @ K_t.transpose(-1, -2) / np.sqrt(tokens.shape[-1])
            scores_t = torch.where(mask_t, scores_t, torch.full_like(scores_t, -1e9))
            attention_output_t = torch.softmax(scores_t, dim=-1) @ V_t
            residual1_t = tokens_t + attention_output_t
            norm1_t = torch.nn.functional.layer_norm(residual1_t, normalized_shape=(tokens.shape[-1],))
            ff_hidden_t = torch.nn.functional.gelu(norm1_t @ W1_t + b1_t, approximate="tanh")
            ff_output_t = ff_hidden_t @ W2_t + b2_t
            block_output_t = torch.nn.functional.layer_norm(norm1_t + ff_output_t, normalized_shape=(tokens.shape[-1],))

            print(torch.allclose(block_output_t, torch.tensor(block_output), atol=1e-6))
            """
        ),
        markdown_cell("## 4. Custom Figure\n\nThe block changes token norms while the attention map shows how much each token borrows from previous positions."),
        code_cell(
            """
            token_norms_before = np.linalg.norm(tokens[0], axis=-1)
            token_norms_after = np.linalg.norm(block_output[0], axis=-1)
            labels = [f"t{i}" for i in range(tokens.shape[1])]

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            heatmap = axes[0].imshow(attention_weights[0], cmap="Blues", vmin=0.0, vmax=1.0)
            axes[0].set_xticks(range(len(labels)), labels)
            axes[0].set_yticks(range(len(labels)), labels)
            axes[0].set_title("Masked attention matrix")

            width = 0.35
            positions = np.arange(len(labels))
            axes[1].bar(positions - width / 2, token_norms_before, width=width, label="before")
            axes[1].bar(positions + width / 2, token_norms_after, width=width, label="after")
            axes[1].set_xticks(positions, labels)
            axes[1].set_title("Token-state norms through the block")
            axes[1].legend()

            fig.colorbar(heatmap, ax=axes[0], fraction=0.046)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Case Study: Why Residual Paths Matter

            If attention temporarily focuses on the wrong token, the residual path still preserves the pre-attention state.
            That makes optimization much easier than forcing every layer to rebuild information from scratch.
            """
        ),
    ])


def diffusion_notebook() -> dict[str, object]:
    return notebook_document([
        markdown_cell("# Diffusion Models Math\n\nPart of **ML Math Applied**."),
        markdown_cell(
            """
            ## 1. Intuition First

            A diffusion model destroys structure gradually with Gaussian noise, then learns to reverse that corruption one step at a time.

            ![Diffusion process](../assets/diagrams/diffusion_forward_reverse.svg)
            """
        ),
        code_cell(BASE_IMPORTS),
        markdown_cell(
            r"""
            ## 2. Forward Process Derivation

            With schedule $\{\beta_t\}_{t=1}^T$, define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
            Repeatedly applying

            $$
            q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t) I)
            $$

            yields the closed form

            $$
            q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I).
            $$

            So one noisy sample can be written directly as

            $$
            x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon,
            \qquad
            \epsilon \sim \mathcal{N}(0, I).
            $$
            """
        ),
        code_cell(
            """
            from src.ml_components import diffusion_forward_process, linear_beta_schedule, predict_clean_from_noise

            betas = linear_beta_schedule(30, beta_start=1e-4, beta_end=0.04)
            clean_point = np.array([[1.2, -0.4]])
            known_noise = np.array([[0.3, -0.2]])
            noisy_point, cache = diffusion_forward_process(clean_point, timestep=18, betas=betas, noise=known_noise)
            recovered_point = predict_clean_from_noise(noisy_point, timestep=18, predicted_noise=known_noise, betas=betas)

            rng = np.random.default_rng(7)
            cluster = np.concatenate(
                [rng.normal(loc=(-1.0, -0.6), scale=0.18, size=(60, 2)), rng.normal(loc=(1.0, 0.8), scale=0.18, size=(60, 2))],
                axis=0,
            )
            noisy_snapshots = {}
            for timestep in (0, 10, 20, 29):
                snapshot_noise = rng.standard_normal(cluster.shape)
                noisy_snapshots[timestep], _ = diffusion_forward_process(cluster, timestep=timestep, betas=betas, noise=snapshot_noise)

            print("noisy point =", noisy_point)
            print("recovered point =", recovered_point)
            """
        ),
        markdown_cell("## 3. PyTorch Verification\n\nThe closed-form reconstruction of `x_0` from `x_t` and the true noise should match exactly."),
        code_cell(
            """
            betas_t = torch.tensor(betas, dtype=torch.float64)
            alpha_bar_t = torch.cumprod(1.0 - betas_t, dim=0)[18]
            clean_t = torch.tensor(clean_point, dtype=torch.float64)
            noise_t = torch.tensor(known_noise, dtype=torch.float64)
            noisy_t = torch.sqrt(alpha_bar_t) * clean_t + torch.sqrt(1.0 - alpha_bar_t) * noise_t
            recovered_t = (noisy_t - torch.sqrt(1.0 - alpha_bar_t) * noise_t) / torch.sqrt(alpha_bar_t)

            print(torch.allclose(noisy_t, torch.tensor(noisy_point), atol=1e-8))
            print(torch.allclose(recovered_t, torch.tensor(recovered_point), atol=1e-8))
            """
        ),
        markdown_cell("## 4. Custom Figures\n\nThe first panel tracks the signal and noise coefficients; the second row shows a 2D dataset being diffused."),
        code_cell(
            """
            alpha_bar = np.cumprod(1.0 - betas)
            signal = np.sqrt(alpha_bar)
            noise = np.sqrt(1.0 - alpha_bar)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axes[0, 0].plot(signal, label="sqrt(alpha_bar_t)", linewidth=2.5)
            axes[0, 0].plot(noise, label="sqrt(1 - alpha_bar_t)", linewidth=2.5)
            axes[0, 0].set_title("Signal-to-noise coefficients")
            axes[0, 0].legend()

            axes[0, 1].axis("off")
            axes[0, 1].text(0.05, 0.75, "As t grows:", fontsize=14)
            axes[0, 1].text(0.05, 0.55, "- signal coefficient shrinks", fontsize=12)
            axes[0, 1].text(0.05, 0.40, "- noise coefficient grows", fontsize=12)
            axes[0, 1].text(0.05, 0.25, "- data becomes close to Gaussian", fontsize=12)

            for axis, timestep in zip(axes[1], (0, 29)):
                points = noisy_snapshots[timestep]
                axis.scatter(points[:, 0], points[:, 1], alpha=0.65, s=18)
                axis.set_title(f"cluster at timestep {timestep}")
                axis.set_xlabel("x1")
                axis.set_ylabel("x2")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Case Study: Why Predict Noise

            In image diffusion models, predicting the residual noise keeps the target distribution stable across time.
            That makes the denoising network easier to train than directly predicting a fully clean sample at every step.
            """
        ),
    ])


def vae_notebook() -> dict[str, object]:
    return notebook_document([
        markdown_cell("# VAE Math\n\nPart of **ML Math Applied**."),
        markdown_cell(
            """
            ## 1. Intuition First

            A variational autoencoder compresses data into a probabilistic latent code and pays a KL penalty when that code drifts too far from a simple prior.

            ![VAE ELBO](../assets/diagrams/vae_elbo.svg)
            """
        ),
        code_cell(BASE_IMPORTS),
        markdown_cell(
            r"""
            ## 2. ELBO and Diagonal-Gaussian KL

            For encoder posterior $q_\phi(z \mid x)$ and decoder likelihood $p_\theta(x \mid z)$:

            $$
            \log p_\theta(x)
            \geq
            \mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)]
            -
            \mathrm{KL}(q_\phi(z \mid x) \| p(z)).
            $$

            If $q_\phi(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$, then

            $$
            \mathrm{KL} = \frac{1}{2} \sum_j \left(\exp(\log \sigma_j^2) + \mu_j^2 - 1 - \log \sigma_j^2\right).
            $$

            Sampling uses the reparameterization trick:

            $$
            z = \mu + \sigma \odot \epsilon,
            \qquad
            \epsilon \sim \mathcal{N}(0, I).
            $$
            """
        ),
        code_cell(
            """
            from matplotlib.patches import Ellipse
            from src.ml_components import reparameterize_gaussian, vae_kl_divergence

            mean = np.array([[1.2, -0.4], [0.2, 0.7], [-0.8, 0.3]])
            log_variance = np.log(np.array([[0.25, 1.4], [1.0, 0.5], [0.7, 0.2]]))
            epsilon = np.array([[0.0, 1.0], [-1.0, 0.5], [0.3, -0.2]])

            latent_samples = reparameterize_gaussian(mean, log_variance, noise=epsilon)
            kl_terms = vae_kl_divergence(mean, log_variance)

            beta_values = np.array([0.5, 1.0, 2.0, 4.0])
            reconstruction_error = np.array([0.32, 0.35, 0.42, 0.57])
            beta_elbo = reconstruction_error + beta_values * kl_terms.mean()

            print("latent samples =\\n", latent_samples)
            print("KL terms =", kl_terms)
            """
        ),
        markdown_cell("## 3. PyTorch Verification\n\nCompare the closed-form KL against the same expression built in PyTorch."),
        code_cell(
            """
            mean_t = torch.tensor(mean, dtype=torch.float64)
            log_variance_t = torch.tensor(log_variance, dtype=torch.float64)
            std_t = torch.exp(0.5 * log_variance_t)
            kl_t = 0.5 * torch.sum(torch.exp(log_variance_t) + mean_t**2 - 1.0 - log_variance_t, dim=-1)
            latent_t = mean_t + std_t * torch.tensor(epsilon, dtype=torch.float64)

            print(torch.allclose(kl_t, torch.tensor(kl_terms), atol=1e-8))
            print(torch.allclose(latent_t, torch.tensor(latent_samples), atol=1e-8))
            """
        ),
        markdown_cell("## 4. Custom Figure\n\nThe ellipse visualizes one posterior covariance and the line plot shows how increasing beta trades reconstruction quality for stronger regularization."),
        code_cell(
            """
            covariance = np.diag(np.exp(log_variance[0]))
            widths = 2.0 * np.sqrt(np.diag(covariance))

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].scatter(latent_samples[:, 0], latent_samples[:, 1], color="tab:blue", label="samples")
            ellipse = Ellipse(xy=mean[0], width=widths[0], height=widths[1], angle=0.0, facecolor="none", edgecolor="tab:red", linewidth=2)
            axes[0].add_patch(ellipse)
            axes[0].scatter(mean[:, 0], mean[:, 1], color="tab:orange", marker="x", s=80, label="means")
            axes[0].set_title("Latent posterior geometry")
            axes[0].set_xlabel("z1")
            axes[0].set_ylabel("z2")
            axes[0].legend()

            axes[1].plot(beta_values, beta_elbo, marker="o", linewidth=2.5, color="tab:purple")
            axes[1].set_title("beta-VAE objective proxy")
            axes[1].set_xlabel("beta")
            axes[1].set_ylabel("reconstruction + beta * KL")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Case Study: beta-VAE Tradeoff

            Larger beta values push the posterior closer to the unit Gaussian prior.
            That usually improves latent structure, but it also makes reconstruction harder because the code is more tightly constrained.
            """
        ),
    ])


CUSTOM_NOTEBOOKS = {
    ("01_linear_algebra", "06_PCA_from_scratch.ipynb"): pca_notebook,
    ("08_ml_math_applied", "01_backpropagation_derived.ipynb"): backpropagation_notebook,
    ("08_ml_math_applied", "02_attention_mechanism.ipynb"): attention_notebook,
    ("08_ml_math_applied", "08_transformer_math_complete.ipynb"): transformer_notebook,
    ("08_ml_math_applied", "09_diffusion_models_math.ipynb"): diffusion_notebook,
    ("08_ml_math_applied", "10_VAE_math.ipynb"): vae_notebook,
}


def notebook(topic_title: str, topic_dir: str, module_name: str | None, filename: str) -> dict[str, object]:
    builder = CUSTOM_NOTEBOOKS.get((topic_dir, filename))
    if builder is not None:
        return builder()
    title = title_from_name(filename)
    intuition, analogy, ml_use = lesson_profile(filename)
    numpy_code, torch_code, viz_code = code_for_topic(topic_dir, module_name)
    return notebook_document([
        markdown_cell(f"# {title}\n\nPart of **{topic_title}**."),
        markdown_cell(f"## 1. Intuition First\n\n{intuition}\n\n**Analogy:** {analogy}\n\n**Why this matters in ML:** {ml_use}"),
        markdown_cell("## 2. Mathematical Foundation\n\nState the object precisely, define every symbol, and derive the working identity line by line instead of jumping to the final formula.\n\n- Write the definition first.\n- Show the intermediate algebra.\n- Explain the assumptions.\n- Flag the common misconception before moving on."),
        code_cell(BASE_IMPORTS),
        markdown_cell("## 3. NumPy Implementation From Scratch\n\nUse the local teaching module or plain NumPy to implement the core idea explicitly."),
        code_cell(numpy_code),
        markdown_cell("## 4. PyTorch Verification\n\nRebuild the same quantity in PyTorch and verify equivalence with `torch.allclose`."),
        code_cell(torch_code),
        markdown_cell("## 5. Visualization\n\nConnect the formula to geometry, distributions, optimization behavior, or spectral structure."),
        code_cell(viz_code),
        code_cell("try:\n    import ipywidgets as widgets\n    widgets.IntSlider(description='Explore', min=0, max=10, value=5)\nexcept Exception as exc:\n    print('ipywidgets is optional for this notebook:', exc)"),
        markdown_cell("## 6. Where This Lives in Real Models\n\n```python\noutput = module(inputs)\nloss = criterion(output, targets)\nloss.backward()\n```\n\nIf the math is wrong here, the implementation drifts even when the code still runs."),
        markdown_cell(f"## 7. Exercises\n\n1. Beginner: compute a tiny hand-worked example for **{title}**.\n2. Intermediate: modify the demo inputs and predict the change before running the cell.\n3. Advanced: rebuild the same computation without using the helper function.\n\n<details>\n<summary>Hidden solution ideas</summary>\n\n- Start from the definition.\n- Check dimensions and edge cases.\n- Verify the result again in PyTorch.\n</details>"),
    ])


def write_notebook(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def topic_readme(title: str, module_name: str | None, notebooks: list[str]) -> str:
    lines = [f"# {title}", "", "## Lessons", ""]
    lines += [f"- [{title_from_name(nb)}](./{nb})" for nb in notebooks]
    lines += ["", "## Start Here", "", "- [Intuition Guide](./intuition_guide.md)"]
    if module_name:
        lines += ["", "## Local Module", "", f"`src/{module_name}.py` re-exports `{IMPORTS[module_name]}` for notebook-local imports."]
    if title == "ML Math Applied":
        lines += [
            "",
            "## Highlights",
            "",
            "- Rich case studies now cover backpropagation, masked attention, transformer blocks, diffusion, and VAEs.",
            "- Reusable diagrams live under `../assets/diagrams/` and are referenced directly from the notebooks.",
        ]
    lines += ["", "## Suggested Order", "", "Work through the numbered notebooks in sequence."]
    return "\n".join(lines)


def root_readme() -> str:
    topic_rows = "\n".join(
        f"| [{title}](./{directory}/README.md) | {len(notebooks)} notebooks |"
        for directory, title, _, notebooks in TOPICS
    )
    return f"""# Mathematics for Machine Learning

[![ci](https://github.com/AKMessi/math-for-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/AKMessi/math-for-ML/actions/workflows/ci.yml)
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

Run notebook execution checks with:

```bash
python scripts/run_notebooks.py
```

## Sections

Each section now includes an `intuition_guide.md` with high-level explanations and visuals before the notebook deep dive.

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


def diagrams_readme() -> str:
    diagram_dir = ROOT / "assets" / "diagrams"
    existing = sorted(path.name for path in diagram_dir.glob("*.svg"))
    lines = ["# Diagrams", "", "Reusable SVG diagrams for notebooks and docs.", "", "## Catalog", ""]
    lines += [f"- [{name}](./{name})" for name in existing]
    return "\n".join(lines)


def generate() -> None:
    write_text(ROOT / "README.md", root_readme())
    write_text(
        ROOT / "CONTRIBUTING.md",
        "# Contributing\n\n## Standards\n\n- Put intuition before equations.\n- Derive results instead of skipping steps.\n- Keep implementations readable, typed, and documented.\n- Verify important calculations against PyTorch.\n- Add or update pytest coverage for source changes.\n\n## Workflow\n\n1. Install dependencies with `python -m pip install -r requirements.txt`.\n2. Install the package with `python -m pip install -e .`.\n3. Run `python -m pytest`.\n4. If you changed notebooks, run `python scripts/run_notebooks.py`.\n",
    )
    write_text(
        ROOT / "ROADMAP.md",
        "# Roadmap\n\n## Done\n\n- [x] Installable `math_for_ml` package\n- [x] Topic scaffolding and 59 notebooks\n- [x] Notebook-local wrapper modules\n- [x] Pytest coverage for the main source modules\n- [x] Cheatsheets and learner guidance\n- [x] Expand more notebooks with longer derivations and custom figures\n- [x] Add notebook execution checks in CI\n- [x] Add more reusable diagrams under `assets/diagrams/`\n- [x] Add richer model-specific case studies in the applied section\n",
    )
    write_text(ROOT / "LICENSE", "MIT License\n\nCopyright (c) 2026 Open Source Contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.")
    write_text(ROOT / "00_how_to_use" / "learning_paths.md", "# Learning Paths\n\n- Beginner: start with linear algebra, calculus, probability, then applied ML math.\n- Practitioner: jump to SVD, matrix calculus, optimization, numerical stability, and applied notebooks.\n- Researcher: read section README files, then prioritize Jacobians, Hessians, Bayesian inference, information theory, spectral graphs, transformers, diffusion, and VAEs.\n- Quick Reference: use `cheatsheets/` and topic README files.\n")
    write_text(ROOT / "00_how_to_use" / "prerequisites.md", "# Prerequisites\n\n- Basic algebra and equation manipulation\n- Basic Python and package installation\n- Willingness to work through code, not just formulas\n\nIf something feels abstract, move to the implementation or visualization section first, then return to the derivation.\n")
    for name, content in CHEATSHEETS.items():
        write_text(ROOT / "cheatsheets" / name, content)
    for name, svg in DIAGRAMS.items():
        write_text(ROOT / "assets" / "diagrams" / name, svg)
    write_text(ROOT / "assets" / "diagrams" / "README.md", diagrams_readme())

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
