from setuptools import setup
from setuptools import find_packages

setup(
    name='qdax',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/adaptive-intelligent-robotics/QDax',
    license='',
    author='Bryan Lim',
    author_email='',
    description='',
    install_requires=[
        "absl-py==1.0.0",
        "jax==0.2.26",
        "brax==0.0.10",
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
