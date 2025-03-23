from setuptools import setup, find_packages

setup(
    name="dspy_optimizer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "dspy_optimizer": ["data"],
    },
    install_requires=[
        "dspy-ai>=2.0.0",
        "pytest>=7.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "dspy-optimize=dspy_optimizer.cli:optimize",
            "dspy-apply=dspy_optimizer.cli:apply_to_app",
            "dspy-cli=dspy_optimizer.cli:main",
        ],
    },
    python_requires=">=3.8",
)