# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diveristy (QD) algorithms through hardware accelerators and massive parallelism. 

QDax paper: https://arxiv.org/abs/2202.01258 

## Installation
```
pip install git+https://github.com/adaptive-intelligent-robotics/QDax.git@main
```

## How to run code
There are two ways to do this.
1. Colab Notebooks (has visualization included) - reccomended (to also avoid needing to download dependencies and configure environment)
Open the notebook qd_run in the notebooks directory and run the notebook according the walkthrough instructions.

2. Locally
A singularity folder is provided to easily install everything in a container. If you use singularity image or install the dependencies locally, you can run a single exeperiment using for example: 
```
python run_qd.py --env_name walker --grid_shape 30 30 --batch_size 2048
```
Alternatively, to run experiments that compare the effect of batch sizes, use command below. For example, to run the experiments on the walker environment (which has a 2-d BD) with a grid shape of (30,30) with 5 replications. 
```
python3 run_comparison_batch_sizes.py --env_name walker --grid_shape 30 30 -n 5
CUDA_VISIBLE_DEVICES=0 python3 run_comparison_batch_sizes.py --env_name walker --grid_shape 30 30 -n 5
CUDA_VISIBLE_DEVICES="0,1" python3 run_comparison_batch_sizes.py --env_name walker --grid_shape 30 30 -n 5
```

### How to run analysis and see plots
Expname is the name of the directories of the experiments (it will look for directory that start with that string. Results is the directory containing all the results folders. Attribute in which we want to compare the results on.
```
python3 analysis/plot_metrics.py --exp-name qdbrax_training --results ./qdax_walker_fixednumevals/ --attribute population_size --save figure.png
```

## Code Structure (for developers)
Some things to note beforehand is that JAX relies on a functional programming paradigm. We will try as much as possible to maintain this programming style.

The main file used is `qdbrax/training/qd.py`. This file contains the main `train` function which consists of the entire QD loop and supporting functions.
- INPUTS: The `train` function takes as input the task, emitter and hyperparameters. 
- FUNCTIONS: The main functions used by `train` are also declared in this file. Working in top_down importance in terms of how the code works. The key function here is the `_es_one_epoch` function. In terms of QD, this determines the loop performed at each generation: Selection (from archive) and Variation to generate solutions to be evaluated defined by the `emitter_fn`, Evaluation and Archive Update defined by (`eval_and_add_fn`). The first part of the `train` function `init_phase_fn` which initializes the archive using random policies.
- ORDER: `train` first calls `init_phase_fn` and then `_es_one_epoch` for defined number of generations or evaluations.

## Notes
### Key Management
```
key = jax.random.PRNGKey(seed)
key, key_model, key_env = jax.random.split(key, 3)
```
- key is for training_state.key
- key_model is for policy_model.init
- key_env is for environment initialisations (although in our deterministic case we do not really use this)

From the flow of the program we perform an init_phase first. The init_phase function uses the training_state.key and outputs the updated training_state (with a new key) after performing the initialization (initailization of archive by evaluating random policies).

After this we depend on the training_state.key in es_one epoch to be managed. In the es_one_epoch(training_state):
```
key, key_emitter, key_es_eval = jax.random.split(training_state.key, 3)
```
- key_selection passed into selection function
- key_petr is passed into mutation function (iso_dd)
- key_es_eval is passed into eval_and_add
- key is saved as the new training_state.key for the next epoch
And the training_state is returned as an output of this function.


