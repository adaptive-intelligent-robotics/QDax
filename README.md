[![Documentation Status](https://readthedocs.org/projects/qdax/badge/?version=latest)](https://qdax.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/adaptive-intelligent-robotics/QDax/actions/workflows/ci.yaml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adaptive-intelligent-robotics/QDax/blob/main/LICENSE)



# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diversity (QD) and neuro-evolution algorithms through hardware accelerators and massive parallelism. QDax has been developed as a research framework: it is flexible and easy to extend.

- QDax [paper](https://arxiv.org/abs/2202.01258)
- QDax [documentation](https://qdax.readthedocs.io/en/latest/)

## Hands on QDax
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/)

To see how QDax works, you can run this [notebook](./notebooks/mapelites_example.ipynb) on colab, it is an example of MAP-Elites evolving a population of controllers on a chosen Brax environment (Walker by default).

## Installation

The simplest way to install QDax is to clone the repo and to install the requirements.

```bash
pip install git+https://github.com/adaptive-intelligent-robotics/QDax.git
cd QDax
pip install -r requirements.txt
```

Nevertheless, we recommand to use either Docker, Singularity or conda to use the repository. Steps to do so are presented in the [documentation](https://qdax.readthedocs.io/en/latest/installation/).

## Basic API Usage
```python

import qdax

```


## QDax Algorithms
- [MAP-Elites](https://arxiv.org/abs/1504.04909)
- [CVT MAP-Elites](https://arxiv.org/abs/1610.05729)
- [Policy Gradient Assisted MAP-Elites](https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf)
- [Differentiable Quality-Diversity (DQD) - OMG-MEGA and CMA-MEGA](https://arxiv.org/abs/2106.03894)
- [Multi-Objective Quality-Diversity (MOME)]()

QDax currently supports the following algorithms.

## QDax flexibility

QDax has been designed to be modular yet flexible so it's easy for anyone to use or extend and build on the different QD algortihms.
For instance, MAP-Elites is designed to work with simple components: a user can hence create a new emitter and pass it to the MAPElites class so he does not have to re-implement the evaluation and addition steps. Similarly, the user can also implement its own container.

QDax has also similarities to simple `ask`/`tell` interface. The `ask` function is similar to the `emit` function. The `tell` function is similar to the `update`. However, most importantly QDax handles the archive management which is the key idea of QD algorihtms and not present/needed in standard optimization algorihtms or evolutionary strategies.


## Contributions
Issues and contributions are welcome. Please this the [documentation](https://qdax.readthedocs.io/en/latest/guides/CONTRIBUTING/) to see how you can contribute to the project.

## Related Projects
- [EvoJax: Hardware-Accelerated Neuroevolution](https://github.com/google/evojax). EvoJAX is a scalable, general purpose, hardware-accelerated neuroevolution toolkit. [Paper](https://arxiv.org/abs/2202.05008)
- [evosax: JAX-Based Evolution Strategies](https://github.com/RobertTLange/evosax)

## Acknowledgements and Citing QDax
If you use QDax in your research and want to cite it in your work, please use:
```
@{}
```

## Contributors

QDax is developed and maintained by the [Adaptive & Intelligent Robotics Lab (AIRL)](https://www.imperial.ac.uk/adaptive-intelligent-robotics/) and [InstaDeep](https://www.instadeep.com/). Thank you to all the contributors from both teams!

<a href="https://github.com/limbryan" title="Bryan Lim"><img src="https://github.com/limbryan.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/maxiallard" title="Maxime Allard"><img src="https://github.com/maxiallard.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Lookatator" title="Luca Grilloti"><img src="https://github.com/Lookatator.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Aneoshun" title="Antoine Cully"><img src="https://github.com/Aneoshun.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/felixchalumeau" title="Felix Chalumeau"><img src="https://github.com/felixchalumeau.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/ranzenTom" title="Thomas Pierrot"><img src="https://github.com/ranzenTom.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Egiob" title="Raphael Boige"><img src="https://github.com/Egiob.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/valentinmace" title="Valentin Mace"><img src="https://github.com/valentinmace.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/GRichard513" title="Guillaume Richard"><img src="https://github.com/GRichard513.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/flajolet" title="Arthur Flajolet"><img src="https://github.com/flajolet.png" height="auto" width="50" style="border-radius:50%"></a>
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
