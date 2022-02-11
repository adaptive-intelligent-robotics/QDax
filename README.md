# QDax: Accelerated Quality-Diversity
QDax is a tool to accelerate Quality-Diveristy (QD) algorithms through hardware accelerators and massive parallelism. 

QDax paper: https://arxiv.org/abs/2202.01258 

## Installation

### Dependencies

In particular, QDax relies on the [JAX](https://github.com/google/jax) and [brax](https://github.com/google/brax) libraries. 
To install all dependencies, you can run the following command:
```bash
pip install -r requirements.txt
```

### Installing QDax

```bash
pip install git+https://github.com/adaptive-intelligent-robotics/QDax.git
```

## Examples

There are two ways to run QDax: 
1. Colab Notebooks (has visualization included) - recommended (to also avoid needing to download dependencies and configure environment)
Open the notebook [notebook](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/Run_QDax_Example_Notebook.ipynb) in the notebooks directory and run it according the walkthrough instructions.

2. Locally - A singularity folder is provided to easily install everything in a container. 
If you use singularity image or install the dependencies locally, you can run a single experiment using for example: 

```
python run_qd.py --env_name walker --grid_shape 30 30 --batch_size 2048 --num-evaluations 1000000
```
Alternatively, to run experiments that compare the effect of batch sizes, use command below.
For example, to run the experiments on the walker environment (which has a 2-dimensional BD) with a grid shape of (30,30) with 5 replications. 
```
python3 run_comparison_batch_sizes.py --env_name walker --grid_shape 30 30 -n 5
CUDA_VISIBLE_DEVICES=0 python3 run_comparison_batch_sizes.py --env_name walker --grid_shape 30 30 -n 5
CUDA_VISIBLE_DEVICES="0,1" python3 run_comparison_batch_sizes.py --env_name walker --grid_shape 30 30 -n 5
```

### Analysis and Plotting Tools

Expname is the name of the directories of the experiments (it will look for directory that start with that string. Results is the directory containing all the results folders.

```
python3 analysis/plot_metrics.py --exp-name qdax_training --results ./qdax_walker_fixednumevals/ --attribute population_size --save figure.png
```
where:
- `--exp-name` is the name of the directories of the experiments (it will look for directory that starts with that string.
- `--results` is the directory containing all the results folders.
- `--attribute`: attribute in which we want to compare the results on.


## Code Structure (for developers)
Some things to note beforehand is that JAX relies on a functional programming paradigm. 
We will try as much as possible to maintain this programming style.

<<<<<<< HEAD
The main file used is `qdbrax/training/qd.py`. 
This file contains the main `train` function which consists of the entire QD loop and supporting functions.
- INPUTS: The `train` function takes as input the task, emitter and hyperparameters. 
- FUNCTIONS: The main functions used by `train` are also declared in this file. 
Working in top_down importance in terms of how the code works. 
The key function here is the `_es_one_epoch` function. 
In terms of QD, this determines the loop performed at each generation: 
  (1) Selection (from archive) and Variation to generate solutions to be evaluated defined by the `emitter_fn`,
  (2) Evaluation 
  and (3) Archive Update defined by (`eval_and_add_fn`). 
The first part of the `train` function calls `init_phase_fn` which initializes the archive using random policies.
- ORDER: `train` first calls `init_phase_fn` and then `_es_one_epoch` for a defined number of generations or evaluations.
=======
The main file used is `qdax/training/qd.py`. This file contains the main `train` function which consists of the entire QD loop and supporting functions.
- Inputs: The `train` function takes as input the task, emitter and hyperparameters. 
- Functions: The main functions used by `train` are also declared in this file. Working in top_down importance in terms of how the code works. The key function here is the `_es_one_epoch` function. In terms of QD, this determines the loop performed at each generation: Selection (from archive) and Variation to generate solutions to be evaluated defined by the `emitter_fn`, Evaluation and Archive Update defined by (`eval_and_add_fn`). The first part of the `train` function is the `init_phase_fn` which initializes the archive using random policies.
- Flow: `train` first calls `init_phase_fn` and then `_es_one_epoch` for defined number of generations or evaluations.

>>>>>>> 3e786643a20ed30814f820e5db7f69539dc37d44

## Notes
### Key Management
```
key = jax.random.PRNGKey(seed)
key, key_model, key_env = jax.random.split(key, 3)
```
- `key` is for training_state.key
- `key_model` is for policy_model.init
- `key_env` is for environment initialisations (although in our deterministic case we do not really use this)

From the flow of the program, we perform an `init_phase` first. 
The `init_phase` function uses the `training_state.key` and outputs the updated `training_state` (with a new key) after performing the initialization (initialization of archive by evaluating random policies).

After this, we depend on the `training_state.key` in `es_one_epoch` to be managed. 
In the `es_one_epoch(training_state)`:
```
key, key_emitter, key_es_eval = jax.random.split(training_state.key, 3)
```
- `key_selection` passed into selection function
- `key_petr` is passed into mutation function (iso_dd)
- `key_es_eval` is passed into `eval_and_add`
- `key` is saved as the new `training_state.key` for the next epoch.
And the `training_state` is returned as an output of this function.

## Contributors

QDax is currently developed and maintained by the [Adaptive & Intelligent Robotics Lab (AIRL)](https://www.imperial.ac.uk/adaptive-intelligent-robotics/):

- [Bryan Lim](https://limbryan.github.io/)
- [Maxime Allard](https://www.imperial.ac.uk/people/m.allard20)
- [Luca Grillotti](https://scholar.google.com/citations?user=gY9CmssAAAAJ&hl=fr&oi=sra)
- [Antoine Cully](https://www.imperial.ac.uk/people/a.cully)



