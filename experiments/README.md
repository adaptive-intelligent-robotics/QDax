This folder contains training scripts to launch algorithms thanks to tools provided by the package. The script `train_mapelites.py` is an example of how to use the package to launch the MAP-Elites algorithm.

The three main phases of this scripts are:
1. Defining util functions, like how to play a step in the env, how to get the bd and fitness.
2. Defining the components of the algorithms: the controllers, the jitted util functions, mutation and crossover functions, the cvt centroids and the mapelites instance.
3. The training/optimization loop, written thanks to the methods defined in the mapelites instance.

By default, `python experiments/train_mapelites.py` will evolve a CVT grid of 1000 max niches containing controllers defined as neural networks with two hidden layers of 64 units on a control task called pointmaze (a maze, were the observation/action size is 2x2).

One can easily change the environment used, as well as all the hyperparameters through the `ExpConfig`. This script can easily be plugged to tools like Hydra to manage configs and/or Neptune/Tensorboard to follow the training process.

In this simple version, one can follow the training process through logs in the console as well as plotted grids that will be put in `exp_outputs/MAP-Elites/pointmaze/date/images/me_grids/`.
