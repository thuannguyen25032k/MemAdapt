import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="embodiedbench",
    version="0.2.0",
    author="Thuan Nguyen",
    description=(
        "MemAdapt: A plug-and-play memory adapter for stale-memory reasoning "
        "in embodied agents"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thuannguyen25032k/MemAdapt",
    packages=setuptools.find_packages(
        exclude=["tests*", "habitat-lab*", "*.egg-info*"]
    ),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "typing_extensions>=4.7.0",
    ],
    extras_require={
        "qlora": ["bitsandbytes>=0.41.0"],
        "tracking": ["wandb>=0.16.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
