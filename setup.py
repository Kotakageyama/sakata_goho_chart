from setuptools import setup, find_packages

setup(
    name="sakata_goho_chart",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "optuna",
        "backtesting",
        "tensorflow",
        "torch"
    ],
    python_requires=">=3.8",
)