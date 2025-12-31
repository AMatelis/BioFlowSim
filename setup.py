from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="BioFlowSim",
    version="0.1.0",
    author="Andrew Matelis",
    author_email="your_email@example.com",
    description="A Python library for simulating 2D particle flow in microfluidic channels, generating realistic imaging sequences, and creating ML-ready datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BioFlowSim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
        "opencv-python>=4.7",
        "torch>=2.0",
        "matplotlib>=3.7",
        "ipywidgets>=8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Optional: define CLI tools if you plan to add them
            # "bioflowsim-cli=bioflowsim.cli:main",
        ],
    },
)
