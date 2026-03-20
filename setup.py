from setuptools import setup, find_packages

setup(
    name="spatial-efficiency-eeg",
    version="1.0.0",
    author="Alastair Waterman",
    description="Spatial efficiency metrics for EEG consciousness research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "mne>=1.0",
        "scikit-learn>=1.0",
        "antropy>=0.1.4",
        "matplotlib>=3.4",
        "pandas>=1.3",
    ],
)
