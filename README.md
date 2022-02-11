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
### Environments
#### Resets and Auto-resets
QD should not work using auto-resets as we need the policy/individual to be representative of one episode. To handle this, we turn off the auto-resets and run the episode to the end. We kill the solution entirely and discard it completely if any of the termination conditions are hit (i.e. falling below a certain z-value).
Another approach might be to use the BD of the timestep right before termination (this is what is done in ME-ES implementation). This approach is also implemented and is used by default.

Technical Implementation: To avoid/stop using the AutoResetWrapper of the brax environments, we can set auto_reset=False when creating the environment. However, we then have to manually reset the environments as nothing else resets the environments for you. 
- This was implemented by placing the resets inside the run_es_eval function. 
- To do this we jit another reset_fn which is not a vmapped (i.e. a non-batch reset_fn) and use this in the run_es_eval

#### ME implementation details to account for difficult environments to init policies
Some environments are very difficult (i.e. humanoid) to initialize the ME algorihtm. The initial random policies all fall down and hit the termination condition. For more difficult environments, even the initial generations find it difficult to complete an episode wihtout termination. To overcome this, we implement ME by considering all individuals even if they fall/or have termination. We consider the BD and fitness before termination for all such indiviudals. Because a survive reward is included. 

Tehnical Implementation: This is implemented by outputting a done trajectory vector and checking the index of the timestep in which the done condition (termination) was first hit. With this index, we can then get the observation trajectory up to this timestep to get the BDs right before termination. For fitness, we could also do the same and sum the rewards at every step up till the termination index to get the correspoding fitness. However, the current implementation does NOT save the entire reward trajectory and implements a more ES type implementation in which reward is just summed in one variable called cumulative_reward. So currently, the check is done in the env itself, where we introduce a new class property called self.done which is set to False by default and at reset. This class property is then set to true at the first instance of a done/termination condition. The reward for each step is then decided based on this class property. Hence, the main loop code keeps its same structure and just ocnitnues summing the rewards as usual.


#### Ant env
Order of observations are:
1. pos - root (x,y,z) or (z)
2. rot - root (w,x,y,z) - 4
3. pos - joints (n_dof) 
4. vel - root (x,y,z) - 3
5. angvel - root (x,y,z) - 4
6. vel - joints (n_dof)
7. cfrc - TODO: TO UNDERSTAND WHAT THE DIMENSIONS ACTUALLY ARE HERE

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
key, key_selection, key_petr, key_es_eval = jax.random.split(training_state.key, 4)
# NEW UPDATE: more general
key, key_emitter, key_es_eval = jax.random.split(training_state.key, 3)
```
- key_selection passed into selection function
- key_petr is passed into mutation function (iso_dd)
- key_es_eval is passed into eval_and_add
- key is saved as the new training_state.key for the next epoch
And the training_state is returned as an output of this function.


