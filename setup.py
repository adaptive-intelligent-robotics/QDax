from setuptools import find_packages, setup

setup(
    name="qdax",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/adaptive-intelligent-robotics/QDax",
    license="",
    author="Bryan Lim and InstaDeep Ltd",
    author_email="",
    description="A Python Library for Quality Diversity and NeuroEvolution",
    install_requires=[
        "absl-py==1.0.0",
        "jax==0.3.2",
        "flax==0.4.1",
        "brax==0.0.12",
        "gym>=0.15",
        "numpy>=1.19",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "sklearn",
    ],
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_releases.html",
    ],
)
