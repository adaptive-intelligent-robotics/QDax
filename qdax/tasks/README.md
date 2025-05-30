# QD Tasks
The `tasks` directory provides default `scoring_function`'s to import easily to perform experiments without the boilerplate code so that the main script is kept simple and is not bloated. It provides a set of fixed tasks that is not meant to be modified. If you are developing and require the flexibility of modifying the task and the details that come along with it, we recommend copying and writing your own custom `scoring_function` in your main script instead of importing from `tasks`.

The `tasks` directory also serves as a way to maintain a QD benchmark task suite that can be easily accessed. We implement several benchmark task across a range of domains. The tasks here are classical tasks from QD literature as well as more recent benchmarks tasks proposed at the [QD Benchmarks Workshop at GECCO 2022](https://quality-diversity.github.io/workshop).

## Arm
| Task      | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|-----------|----------------------|------------------|-----------------------|-------------------|-------------|
| Arm       | n                    | $[0,1]^n$        | 2                     | $[0,1]^2$         |             |
| Noisy Arm | n                    | $[0,1]^n$        | 2                     | $[0,1]^2$         |             |

Notes:
- the parameter space is normalized between $[0,1]$ which corresponds to $[0,2\pi]$
- the descriptor space (end-effector x-y position) is normalized between $[0,1]$

### Example Usage

```python
import jax
from qdax.tasks.arm import arm_scoring_function

key = jax.random.key(0)

# Get scoring function
scoring_fn = arm_scoring_function

# Get Task Properties (parameter space, descriptor space, etc.)
min_param, max_param = 0., 1.
min_desc, max_desc = 0., 1.

# Get initial batch of parameters
num_param_dimensions = ...
init_batch_size = ...
key, _subkey = jax.random.split(key)
initial_params = jax.random.uniform(
    _subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Get number of descriptor dimensions
desc_size = 2
```

## Standard Functions
| Task                 | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|----------------------|----------------------|------------------|-----------------------|-------------------|-------------|
| Sphere               | n                    | $[0,1]^n$        | 2                     | $[0,1]^n$         |             |
| Rastrigin            | n                    | $[0,1]^n$        | 2                     | $[0,1]^n$         |             |
| Rastrigin-Projection | n                    | $[0,1]^n$        | 2                     | $[0,1]^n$         |             |

### Example Usage

```python
import jax
from qdax.tasks.standard_functions import sphere_scoring_function

key = jax.random.key(0)

# Get scoring function
scoring_fn = sphere_scoring_function

# Get Task Properties (parameter space, descriptor space, etc.)
min_param, max_param = 0., 1.
min_desc, max_desc = 0., 1.

# Get initial batch of parameters
num_param_dimensions = ...
init_batch_size = ...
key, _subkey = jax.random.split(key)
initial_params = jax.random.uniform(
    _subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Get number of descriptor dimensions
desc_size = 2
```

## Hyper-Volume Functions
"Hypervolume-based Benchmark Functions for Quality Diversity Algorithms" by Jean-Baptiste Mouret

| Task                  | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|-----------------------|----------------------|------------------|-----------------------|-------------------|-------------|
| Square                | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Checkered             | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Empty Circle          | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Non-continuous Islands | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Continuous Islands     | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |

### Example Usage

```python
import jax
from qdax.tasks.hypervolume_functions import square_scoring_function

key = jax.random.key(0)

# Get scoring function
scoring_fn = square_scoring_function

# Get Task Properties (parameter space, descriptor space, etc.)
min_param, max_param = 0., 1.
min_desc, max_desc = 0., 1.

# Get initial batch of parameters
num_param_dimensions = ...
init_batch_size = ...
key, _subkey = jax.random.split(key)
initial_params = jax.random.uniform(
    _subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Get number of descriptor dimensions
desc_size = num_param_dimensions
```

## QD Suite
"Towards QD-suite: developing a set of benchmarks for Quality-Diversity algorithms" by Achkan Salehi and Stephane Doncieux

| Task                           | Parameter Dimensions | Parameter Bounds                                                               | Descriptor Dimensions                 | Descriptor Bounds                                                           | Description |
|--------------------------------|----------------------|--------------------------------------------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------|-------------|
| archimedean-spiral-v0          | 1                    | $[0,\alpha\pi]^n$ (angle param.)<br/> $[0,max-arc-length]$ (arc length param.) | 1 (geodesic descriptor)<br/> 2 (euclidean descriptor) | $[0,max-arc-length]$ (geodesic descriptor)<br/> $[-radius,radius]^2$ (euclidean descriptor) |             |
| SSF-v0                         | $n$                  | Unbounded                                                                      | 1                                     | $[ 0 ,$ ∞ $)$                                                               |             |
| deceptive-evolvability-v0<br/> | $n$ (2 by default)   | Bounded area including the two gaussian peaks                                  | 1                                     | $[0,max-sum-gaussians]$                                                     |             |

### Example Usage

```python
import math
from qdax.tasks.qd_suite import archimedean_spiral_v0_angle_euclidean_task

task = archimedean_spiral_v0_angle_euclidean_task

# Get scoring function
scoring_fn = task.scoring_function

# Get Task Properties (parameter space, descriptor space, etc.)
min_param, max_param = task.get_min_max_params()
min_desc, max_desc = task.get_bounded_min_max_descriptor()  # To consider bounded Descriptor space
# If the task has a descriptor space that is not bounded, then the unbounded descriptor
# space can be obtained via the following:
# min_descriptor, max_descriptor = task.get_min_max_descriptor()

# Get initial batch of parameters
initial_params = task.get_initial_parameters(batch_size=...)

# Get number of descriptor dimensions
desc_size = task.get_descriptor_size()
```

## Brax-RL
QDax provides organized utilities for working with Brax environments in the context of Quality Diversity. The Brax integrations are now structured in a versioned manner to support both Brax v1 and v2 APIs:

- `tasks.brax.v1`: Contains code compatible with Brax v1 API
- `tasks.brax.v2`: Contains code compatible with Brax v2 API

| Task            | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|-----------------|----------------------|------------------|-----------------------|-------------------|-------------|
| pointmaze       | NN params            | Unbounded        | 2                     | $[-1,1]^2$        |             |
| hopper_uni      | NN params            | Unbounded        | 1                     | $[0,1]$           |             |
| walker2d_uni    | NN params            | Unbounded        | 2                     | $[0,1]^2$         |             |
| halfcheetah_uni | NN params            | Unbounded        | 2                     | $[0,1]^2$         |             |
| ant_uni         | NN params            | Unbounded        | 4                     | $[0,1]^4$         |             |
| humanoid_uni    | NN params            | Unbounded        | 2                     | $[0,1]^2$         |             |
| ant_omni        | NN params            | Unbounded        | 2                     | $[-30,30]^2$      |             |
| humanoid_omni   | NN params            | Unbounded        | 2                     | $[-30,30]^2$      |             |
| anttrap         | NN params            | Unbounded        | 2                     | $[-8,8]\times[0,30]$   |             |
| antmaze         | NN params            | Unbounded        | 2                     | $[-5,40]\times[-5,40]$ |             |

Notes:
- The parameter dimensions for default Brax-RL tasks depend on the size and architecture of the neural network used and can be customized and changed easily. If not set, a network size of two hidden layers of size 64 is used.
- The Brax environments are now properly versioned to support both Brax v1 and v2 APIs.

### Example Usage

For Brax v1:
```python
import jax
import qdax.tasks.brax.v1 as environments
from qdax.tasks.brax.v1.env_creators import create_default_brax_task_components

# Create environment, policy network and scoring function
env_name = "ant_omni"
key = jax.random.key(0)
env, policy_network, scoring_fn = create_default_brax_task_components(
    env_name=env_name,
    key=key,
    episode_length=100
)
```

For Brax v2:
```python
import jax
import qdax.tasks.brax.v2 as environments
from qdax.tasks.brax.v2.env_creators import create_default_brax_task_components

# Create environment, policy network and scoring function
env_name = "ant_omni"
key = jax.random.key(0)
env, policy_network, scoring_fn = create_default_brax_task_components(
    env_name=env_name,
    key=key,
    episode_length=100
)
```

See [Brax v1](../../examples/mapelites.ipynb) and [Brax v2](../../examples/mapelites_brax_v2.ipynb) notebooks for more detailed examples.

## Jumanji-RL

QDax provide utils to interact easily with the suite of environments implemented in [Jumanji](https://github.com/instadeepai/jumanji).

### Example Usage

See [Example in Notebook](../../examples/jumanji_snake.ipynb)
