"""Setup script for Raman-Physics-AI MVP.

Usage:
    pip install -e .            # editable install (recommended for development)
    pip install .               # standard install
"""
from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="raman-physics-ai",
    version="0.1.0.dev0",
    description=(
        "Physics-informed deep learning for Raman spectroscopy: "
        "composition prediction, bond identification, OOD detection."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Raman-Physics-AI Contributors",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*", "engine", "engine.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "torch>=2.0.0,<3.0.0",
        "lmfit>=1.2.0,<2.0.0",
        "pybaselines>=1.0.0,<2.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "pyyaml>=6.0.0,<7.0.0",
        "tqdm>=4.65.0,<5.0.0",
    ],
    extras_require={
        "dev":  ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ipykernel>=6.25.0"],
        "viz":  ["seaborn>=0.12.0", "plotly>=5.15.0"],
        "dash": ["streamlit>=1.28.0"],
        "trk":  ["wandb>=0.15.0", "tensorboard>=2.13.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
