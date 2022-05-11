[![Documentation Status](https://readthedocs.org/projects/qdax/badge/?version=latest)](https://qdax.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/instadeepai/QDax/actions/workflows/ci.yaml/badge.svg?branch=2-instadeep-new-structure-suggestion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/instadeepai/QDax/blob/2-instadeep-new-structure-suggestion/LICENSE)



# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diveristy (QD) and neuro-evolution algorithms through hardware accelerators and massive parallelism. QDax has been developped as a research framework: it is flexible and easy to extend.

QDax paper: https://arxiv.org/abs/2202.01258
QDax documentation: https://qdax.readthedocs.io/en/latest/

## Hands on QDax
To see how QDax works, you can run this [notebook](./notebooks/mapelites_example.ipynb) on colab, it is an example of MAP-Elites evolving a population of controllers on a chosen Brax environment (Walker by default).

## Installation

The simplest way to install QDax is to clone the repo and to install the requirements.

```bash
pip install git+https://github.com/adaptive-intelligent-robotics/QDax.git
cd QDax
pip install -r requirements.txt
```

Nevertheless, we recommand to use either Docker, Singularity or conda to use the repository. Steps to do so are presented in the [documentation](https://qdax.readthedocs.io/en/latest/installation/).

## Current components of QDax

### Algorithms
- MAP-Elites
- CVT MAP-Elites
- Policy Gradient Assisted MAP-Elites

### Tasks and environments
- Brax environments with wrappers to access feet contact and xy position of the agent in the environment.
- AntTrap
- AntMaze
- PointMaze

## QDax flexibility

QDax has been designed to be flexible so it's easy for anyone to extend it. For instance, MAP-Elites is designed to work with many different components: a user can hence create a new emitter and pass it to the MAPElites class without having to re-implement everything.

## Contributions
Issues and contributions are welcome. Please this the [documentation]() to see how you can contribute to the project.

## Related Projects
- [EvoJax: Hardware-Accelerated Neuroevolution](https://github.com/google/evojax). EvoJAX is a scalable, general purpose, hardware-accelerated neuroevolution toolkit. [Paper](https://arxiv.org/abs/2202.05008)


## Contributors

QDax is currently developed and maintained by the [Adaptive & Intelligent Robotics Lab (AIRL)](https://www.imperial.ac.uk/adaptive-intelligent-robotics/):

- [Bryan Lim](https://limbryan.github.io/)
- [Maxime Allard](https://www.imperial.ac.uk/people/m.allard20)
- [Luca Grillotti](https://scholar.google.com/citations?user=gY9CmssAAAAJ&hl=fr&oi=sra)
- [Antoine Cully](https://www.imperial.ac.uk/people/a.cully)
