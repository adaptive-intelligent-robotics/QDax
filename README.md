[![Documentation Status](https://readthedocs.org/projects/qdax/badge/?version=latest)](https://qdax.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/instadeepai/QDax/actions/workflows/ci.yaml/badge.svg?branch=2-instadeep-new-structure-suggestion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/instadeepai/QDax/blob/2-instadeep-new-structure-suggestion/LICENSE)



# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diveristy (QD) and neuro-evolution algorithms through hardware accelerators and massive parallelism. QDax has been developped as a research framework: it is flexible and easy to extend.

QDax paper: https://arxiv.org/abs/2202.01258
QDax documentation: https://qdax.readthedocs.io/en/latest/

## Hands on QDax
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/instadeepai/QDax/blob/2-instadeep-new-structure-suggestion/)

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

QDax is currently developed and maintained by the [Adaptive & Intelligent Robotics Lab (AIRL)](https://www.imperial.ac.uk/adaptive-intelligent-robotics/) and [InstaDeep](https://www.instadeep.com/). Thank you to all the contributors from both teams!

From the AIRL team:
<a href="https://github.com/limbryan" title="Bryan Lim"><img src="https://github.com/limbryan.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/maxiallard" title="Maxime Allard"><img src="https://github.com/maxiallard.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Lookatator" title="Luca Grilloti"><img src="https://github.com/Lookatator.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Aneoshun" title="Antoine Cully"><img src="https://github.com/Aneoshun.png" height="auto" width="50" style="border-radius:50%"></a> & From the InstaDeep team:
<a href="https://github.com/felixchalumeau" title="Felix Chalumeau"><img src="https://github.com/felixchalumeau.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/ranzenTom" title="Thomas Pierrot"><img src="https://github.com/ranzenTom.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Egiob" title="Raphael Boige"><img src="https://github.com/Egiob.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/valentinmace" title="Valentin Mace"><img src="https://github.com/valentinmace.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/remidebette" title="Rémi Debette"><img src="https://github.com/remidebette.png" height="auto" width="50" style="border-radius:50%"></a>

<!-- - [Felix Chalumeau](https://www.linkedin.com/in/f%C3%A9lix-chalumeau-083457172/)
- [Thomas Pierrot](https://scholar.google.fr/citations?user=0zBiyNUAAAAJ&hl=en)
- [Raphaël Boige](https://www.linkedin.com/in/raphaelboige/)
- [Valentin Macé](https://www.linkedin.com/in/valentinmace/)
- [Arthur Flajolet](https://scholar.google.com/citations?user=YYwquKkAAAAJ&hl=en)
- [Guillaume Richard](https://scholar.google.com/citations?user=viOjnmQAAAAJ&hl=fr)
- [Rémi Debette](https://www.linkedin.com/in/remidebette/) -->
<!-- - [Bryan Lim](https://limbryan.github.io/)
- [Maxime Allard](https://www.imperial.ac.uk/people/m.allard20)
- [Luca Grillotti](https://scholar.google.com/citations?user=gY9CmssAAAAJ&hl=fr&oi=sra)
- [Antoine Cully](https://www.imperial.ac.uk/people/a.cully) -->
