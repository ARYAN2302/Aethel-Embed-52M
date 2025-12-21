from setuptools import find_packages, setup

setup(
    name="aethel",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "examples", "checkpoints")),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "sentence-transformers>=2.7.0",
        "accelerate>=0.29.0",
    ],
    python_requires=">=3.9",
    description="Long-context, memory-augmented hybrid embeddings",
)
