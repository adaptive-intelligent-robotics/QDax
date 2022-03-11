This folder contains all the necessary elements to write a MAP-Elites training script. (The files were gathered in the folder `algorithms` to isolate them from the rest of QDax)

1. The folder `brax_envs` contains wrappers to get state descriptors from the environments, enabling to easily contruct the behavior descriptors at the end of an episode. It also contains the environment `pointmaze`, a simpler task, using brax api for simplicity but without any physical simulation. Alltogether, this provides `pointmaze`, `anttrap`, `antmaze`, `ant_omni`, `humanoid_omni`, `ant_uni`, `walker2d_uni`, `hopper_uni`, `halfcheetah_uni`, `humanoid_uni`.

2. The file `mutation_operators.py` provides functions to mutation batches of genotypes (it includes mutations and crossovers).

3. The file `types.py` contains types used in this repo. Its principal aim is to ease understanding and clearness of the code, as those are used to type inputs and outputs of functions defined in the repo.

4. The file `utils.py` contains a few util functions that have been gathered here for simplicity.

5. The file `plotting.py` provide a function to plot the MAP-Elites grids.

6. The file `map_elites.py` provides the implementation of the `MAPElitesGrid`, the container used in MAP-Elites. As well as the class `MAPElites`, that gather the few basic steps of the MAP-Elites algorithm, making it easy to compose them in a training script.

One of our design choice is to let all the training scripts outside of the package, hence there is no `train` function in the `MAPElites` class. The training scripts are to be written outside the package. An example is presented in `QDax/experiments/train_mapelites.py`.
