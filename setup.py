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
        "brax>=0.10.4",
        "chex>=0.1.86",
        "flax>=0.8.5",
        "gym>=0.26.2",
        "jax>=0.4.28",
        "jaxlib>=0.4.28",  # necessary to build the doc atm
        "jinja2>=3.1.4",
        "jumanji>=0.3.1",
        "numpy>=1.26.4",
        "optax>=0.1.9",
        "scikit-learn>=1.5.1",
        "scipy>=1.10.1",
        "tensorflow-probability>=0.24.0",
    ],
    extras_require={
        "cuda12": ["jax[cuda12]>=0.4.28"],
    },
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
