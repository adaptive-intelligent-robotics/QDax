[![Documentation Status](https://readthedocs.org/projects/qdax/badge/?version=latest)](https://qdax.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/adaptive-intelligent-robotics/QDax/actions/workflows/ci.yaml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adaptive-intelligent-robotics/QDax/blob/main/LICENSE)



# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diversity (QD) and neuro-evolution algorithms through hardware accelerators and massive parallelism. QD algorithms usually take days/weeks to run on large CPU clusters. With QDax, QD algrotihms can now be run in minutes! ⏩ ⏩ 🕛

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
pip install git+https://github.com/adaptive-intelligent-robotics/QDax.git
```

However, we also provide and recommend using either Docker, Singularity or conda environments to use the repository. Detailed steps to do so are available in the [documentation](https://qdax.readthedocs.io/en/latest/installation/).

## Basic API Usage
For a full and interactive example to see how QDax works, we recommend starting with the tutorial-style [Colab notebook](./notebooks/mapelites_example.ipynb). It is an example of the MAP-Elites algorithm used to evolve a population of controllers on a chosen Brax environment (Walker by default).

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
    (repertoire, _, metrics, random_key,) = map_elites.update(
            repertoire,
            None,
            random_key,
        )

# Get contents of repertoire
repertoire.genotypes, repertoire.fitnesses, repertoire.descriptors
```



## QDax Algorithms
QDax currently supports the following algorithms.
| Algorithm  | Example |
| --- | --- |
| [MAP-Elites](https://arxiv.org/abs/1504.04909) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mapelites_example.ipynb) |
| [CVT MAP-Elites](https://arxiv.org/abs/1610.05729) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mapelites_example.ipynb) |
| [Policy Gradient Assisted MAP-Elites (PGA-ME)](https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/pgame_example.ipynb) |
| [OMG-MEGA](https://arxiv.org/abs/2106.03894) |  [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/omgmega_example.ipynb) |
| [CMA-MEGA](https://arxiv.org/abs/2106.03894) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/cmamega_example.ipynb) |
| [Multi-Objective Quality-Diversity (MOME)](https://arxiv.org/abs/2202.03057) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/mome_example.ipynb) |

## QDax flexibility

QDax has been designed to be modular yet flexible so it's easy for anyone to use and extend on the different state-of-the-art QD algortihms available.
For instance, MAP-Elites is designed to work with a few modular and simple components: `container`, `emitter`, and `scoring function`.

The `container` specifies the structure of archive of solutions to keep and the addition conditions associated with the archive.
The `emitter` component is responsible for generating new solutions to be evaluated. For example, new solutions can be generated with random mutations, gradient descent, or sampling from distributions.
The `scoring function` defines the problem/task we want to solve and functions to evaluate the solutions.

With this modularity, a user can create a new emitter and pass it to the MAPElites class so he does not have to re-implement the evaluation and addition steps. Similarly, users can also implement their own container.

Under one layer of abstraction, users have abit more flexibility. QDax has similarities to the simple and commonly found `ask`/`tell` interface. The `ask` function is similar to the `emit` function. The `tell` function is similar to the `update`. However, most importantly QDax handles the archive management which is the key idea of QD algorihtms and not present/needed in standard optimization algorihtms or evolutionary strategies.

```python
# generate new population with the emitter
genotypes, random_key = self._emitter.emit(
    repertoire, emitter_state, random_key
)

# scores/evaluates the population
fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
    genotypes, random_key
)

# update repertoire
repertoire = repertoire.add(genotypes, descriptors, fitnesses)

# update emitter state after scoring is made
emitter_state = self._emitter.state_update(
    emitter_state=emitter_state,
    repertoire=repertoire,
    genotypes=genotypes,
    fitnesses=fitnesses,
    descriptors=descriptors,
    extra_scores=extra_scores,
)
```


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

<img src="./docs/images/AIRL_Logo.png" alt="AIRL_Logo" width="220"/>
<img src="./docs/images/AIRL_Logo.png" alt="InstaDeep_Logo" width="220"/>

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
