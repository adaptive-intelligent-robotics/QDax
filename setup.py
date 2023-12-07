import os

from setuptools import find_packages, setup

from qdax import __version__

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qdax",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/adaptive-intelligent-robotics/QDax",
    license="MIT",
    author="AIRL and InstaDeep Ltd",
    author_email="adaptive.intelligent.robotics@gmail.com",
    description="A Python Library for Quality-Diversity and NeuroEvolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "absl-py>=1.0.0",
        "jax>=0.4.16",
        "jaxlib>=0.4.16",  # necessary to build the doc atm
        "jinja2<3.1.0",
        "jumanji>=0.3.1",
        "flax>=0.7.4",
        "chex>=0.1.83",
        "brax>=0.9.2",
        "gym>=0.23.1",
        "numpy>=1.22.3",
        "optax>=0.1.7",
        "scikit-learn>=1.0.2",
        "scipy>=1.8.0",
    ],
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_releases.html",
    ],
    keywords=["Quality-Diversity", "NeuroEvolution", "Reinforcement Learning", "JAX"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
