[![Documentation Status](https://readthedocs.org/projects/qdax/badge/?version=latest)](https://qdax.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/adaptive-intelligent-robotics/QDax/actions/workflows/ci.yaml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adaptive-intelligent-robotics/QDax/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/adaptive-intelligent-robotics/QDax/branch/feat/add-codecov/graph/badge.svg)](https://codecov.io/gh/adaptive-intelligent-robotics/QDax)


# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diversity (QD) and neuro-evolution algorithms through hardware accelerators and massive parallelization. QD algorithms usually take days/weeks to run on large CPU clusters. With QDax, QD algorithms can now be run in minutes! ⏩ ⏩ 🕛

QDax has been developed as a research framework: it is flexible and easy to extend and build on and can be used for any problem setting. Get started with simple example and run a QD algorithm in minutes here! [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mapelites_example.ipynb)

- QDax [paper](https://arxiv.org/abs/2202.01258)
- QDax [documentation](https://qdax.readthedocs.io/en/latest/)


## Installation

The latest stable release of QDax can be installed directly from PyPI with:
```bash
pip install qdax
```
Alternatively, to install QDax from source:
```bash
pip install git+https://github.com/adaptive-intelligent-robotics/QDax.git@main
```

However, we also provide and recommend using either Docker, Singularity or conda environments to use the repository. Detailed steps to do so are available in the [documentation](https://qdax.readthedocs.io/en/latest/installation/).

## Basic API Usage
For a full and interactive example to see how QDax works, we recommend starting with the tutorial-style [Colab notebook](./notebooks/mapelites_example.ipynb). It is an example of the MAP-Elites algorithm used to evolve a population of controllers on a chosen Brax environment (Walker by default).

However, a summary of the main API usage is provided below:
```python
import qdax
from qdax.core.map_elites import MAPElites

# Instantiate MAP-Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=mixing_emitter,
    metrics_function=metrics_function,
)

# Initializes repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init(init_variables, centroids, random_key)

# Run MAP-Elites loop
for i in range(num_iterations):
    (repertoire, emitter_state, metrics, random_key,) = map_elites.update(
            repertoire,
            emitter_state,
            random_key,
        )

# Get contents of repertoire
repertoire.genotypes, repertoire.fitnesses, repertoire.descriptors
```


## QDax core algorithms
QDax currently supports the following algorithms:
| Algorithm  | Example |
| --- | --- |
| [MAP-Elites](https://arxiv.org/abs/1504.04909) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mapelites_example.ipynb) |
| [CVT MAP-Elites](https://arxiv.org/abs/1610.05729) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mapelites_example.ipynb) |
| [Policy Gradient Assisted MAP-Elites (PGA-ME)](https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/pgame_example.ipynb) |
| [OMG-MEGA](https://arxiv.org/abs/2106.03894) |  [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/omgmega_example.ipynb) |
| [CMA-MEGA](https://arxiv.org/abs/2106.03894) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/cmamega_example.ipynb) |
| [Multi-Objective Quality-Diversity (MOME)](https://arxiv.org/abs/2202.03057) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mome_example.ipynb) |

## QDax baseline algorithms
The QDax library also provides implementations for some useful baseline algorithms:
| Algorithm  | Example |
| --- | --- |
| [DIAYN](https://arxiv.org/abs/1802.06070) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/diayn_example.ipynb) |
| [DADS](https://arxiv.org/abs/1907.01657) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/dads_example.ipynb) |
| [SMERL](https://arxiv.org/abs/2010.14484) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/smerl_example.ipynb) |


## QDax Overview

QDax has been designed to be modular yet flexible so it's easy for anyone to use and extend on the different state-of-the-art QD algortihms available.
For instance, MAP-Elites is designed to work with a few modular and simple components: `container`, `emitter`, and `scoring_function`.

The `container` specifies the structure of archive of solutions to keep and the addition conditions associated with the archive.

The `emitter` component is responsible for generating new solutions to be evaluated. For example, new solutions can be generated with random mutations, gradient descent, or sampling from distributions as in evolutionary strategies.

The `scoring_function` defines the problem/task we want to solve and functions to evaluate the solutions. For example, the `scoring_function` can be used to represent standard black-box optimization tasks such as rastrigin or RL tasks.

With this modularity, a user can easily swap out any one of the components and pass it to the `MAPElites` class, avoiding having to re-implement all the steps of the algorithm.

Under one layer of abstraction, users have a bit more flexibility. QDax has similarities to the simple and commonly found `ask`/`tell` interface. The `ask` function is similar to the `emit` function in QDax and the `tell` function is similar to the `update` function in QDax. Likewise, the `eval` of solutions is analogous to the `scoring function` in QDax.
More importantly, QDax handles the archive management which is the key idea of QD algorihtms and not present or needed in standard optimization algorihtms or evolutionary strategies.

```python
# Initializes repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init(init_variables, centroids, random_key)

for i in range(num_iterations):

    # generate new population with the emitter
    genotypes, random_key = map_elites._emitter.emit(
        repertoire, emitter_state, random_key
    )

    # scores/evaluates the population
    fitnesses, descriptors, extra_scores, random_key = map_elites._scoring_function(
        genotypes, random_key
    )

    # update repertoire
    repertoire = repertoire.add(genotypes, descriptors, fitnesses)

    # update emitter state
    emitter_state = map_elites._emitter.state_update(
        emitter_state=emitter_state,
        repertoire=repertoire,
        genotypes=genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        extra_scores=extra_scores,
    )
```


## Contributing
Issues and contributions are welcome. Please refer to the [contribution guide](https://qdax.readthedocs.io/en/latest/guides/CONTRIBUTING/) in the documentation for more details.

## Related Projects
- [EvoJAX: Hardware-Accelerated Neuroevolution](https://github.com/google/evojax). EvoJAX is a scalable, general purpose, hardware-accelerated neuroevolution toolkit. [Paper](https://arxiv.org/abs/2202.05008)
- [evosax: JAX-Based Evolution Strategies](https://github.com/RobertTLange/evosax)

## Citing QDax
If you use QDax in your research and want to cite it in your work, please use:
```
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}
```

## Contributors

QDax was developed and is maintained by the [Adaptive & Intelligent Robotics Lab (AIRL)](https://www.imperial.ac.uk/adaptive-intelligent-robotics/) and [InstaDeep](https://www.instadeep.com/).

<img align="center" src="./docs/images/AIRL_logo.png" alt="AIRL_Logo" width="220"/> <img align="center" src="./docs/images/instadeep_logo.png" alt="InstaDeep_Logo" width="220"/>

<a href="https://github.com/limbryan" title="Bryan Lim"><img src="https://github.com/limbryan.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/maxiallard" title="Maxime Allard"><img src="https://github.com/maxiallard.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Lookatator" title="Luca Grilloti"><img src="https://github.com/Lookatator.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/manon-but-yes" title="Manon Flageat"><img src="https://github.com/manon-but-yes.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Aneoshun" title="Antoine Cully"><img src="https://github.com/Aneoshun.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/felixchalumeau" title="Felix Chalumeau"><img src="https://github.com/felixchalumeau.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/ranzenTom" title="Thomas Pierrot"><img src="https://github.com/ranzenTom.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Egiob" title="Raphael Boige"><img src="https://github.com/Egiob.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/valentinmace" title="Valentin Mace"><img src="https://github.com/valentinmace.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/GRichard513" title="Guillaume Richard"><img src="https://github.com/GRichard513.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/flajolet" title="Arthur Flajolet"><img src="https://github.com/flajolet.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/remidebette" title="Rémi Debette"><img src="https://github.com/remidebette.png" height="auto" width="50" style="border-radius:50%"></a>
