from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-ml",
    version="1.0.0",
    author="M.Tech Student",
    author_email="student@university.edu",
    description="Medical Machine Learning for M.Tech Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medical-ml-project",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "medical": [
            "pydicom>=2.3.0",
            "SimpleITK>=2.1.0",
            "nibabel>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-ml=src.mlflow_manager:main",
            "mtech-experiment=src.experiments.medical_ml_experiment:main",
        ],
    },
)