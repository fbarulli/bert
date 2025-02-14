# code is being run on colab, no hardcoded paths are to be used, everything is in the same local dir

from setuptools import setup, find_packages
import os

setup(
    name="simpler_fine_bert",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "wandb>=0.12.0",
        "optuna>=3.0.0",  # Updated for latest features
        "pyyaml>=5.4.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "psutil>=5.8.0",
        "tensorboard>=2.5.0",  # For additional logging
        "scipy>=1.7.0",  # Required by scikit-learn
        "nltk>=3.6.0",  # For text processing
        "datasets>=2.0.0",  # For data handling
        "filelock>=3.0.0"  # Add this line
    ],
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simplified BERT finetuning package with MLM and classification stages",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simpler_fine_bert",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
