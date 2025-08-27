from setuptools import setup, find_packages
from pathlib import Path

def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "Professional ML Engineering Toolkit - Comprehensive measurement suite for ML experiments."

def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    else:
        return [
            "numpy>=1.20.0",
            "pandas>=1.3.0", 
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pyyaml>=5.4.0",
            "codecarbon>=2.1.0",
            "wandb>=0.12.0",
            "tensorboard>=2.8.0"
        ]

setup(
    name="ml-cookbook",
    version="1.0.0",
    author="Daniel Mercado",
    author_email="daniel.mercado@example.com",
    description="Professional ML Engineering Toolkit - Comprehensive measurement suite for ML experiments",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/DanielMercado60660/ml-cookbook",

    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements(),

    entry_points={
        "console_scripts": [
            "cookbook-prof=cookbook.measure.cli:main_cli",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    keywords=[
        "machine-learning", 
        "ml-engineering", 
        "experiment-tracking",
        "performance-profiling",
        "statistical-validation",
        "carbon-tracking",
        "mlops"
    ],

    zip_safe=False,
    platforms=["any"],
)
