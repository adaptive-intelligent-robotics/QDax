from setuptools import find_packages, setup

from qdax import __version__

setup(
    name="qdax",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/adaptive-intelligent-robotics/QDax",
    license="MIT",
    author="Bryan Lim and InstaDeep Ltd",
    author_email="",
    description="A Python Library for Quality Diversity and NeuroEvolution",
    install_requires=[
        "absl-py==1.0.0",
        "jax==0.3.10",
        "jaxlib==0.3.10",  # necessary to build the doc atm
        "flax==0.4.1",
        "brax==0.0.12",
        "gym==0.23.1",
        "numpy==1.22.3",
        "scikit-learn==1.0.2",
        "scipy==1.8.0",
        "sklearn",
    ],
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_releases.html",
    ],
    keywords="Quality Diversity NeuroEvolution Reinforcement Learning JAX",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
