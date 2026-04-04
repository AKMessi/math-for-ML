"""Setuptools configuration for the Mathematics for Machine Learning package."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


setup(
    name="math-for-ml",
    version="0.1.0",
    description="A structured mathematics for machine learning curriculum with tested NumPy implementations.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Open Source Contributors",
    url="https://github.com/AKMessi/math-for-ML",
    license="MIT",
    packages=find_packages(exclude=("*.tests", "*.tests.*", "tests*")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.4.2",
        "scipy==1.17.1",
        "torch==2.10.0",
        "matplotlib==3.10.8",
    ],
    extras_require={
        "dev": [
            "ipywidgets==8.1.7",
            "jupyterlab==4.4.3",
            "nbclient==0.10.2",
            "nbformat==5.10.4",
            "pytest==8.4.2",
        ]
    },
    project_urls={
        "Source": "https://github.com/AKMessi/math-for-ML",
        "Tracker": "https://github.com/AKMessi/math-for-ML/issues",
    },
)
