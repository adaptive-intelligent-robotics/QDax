# QD Tasks
The `tasks` directory provides default `scoring_function`'s to import easily to perform experiments without the boilerplate code so that the main script is kept simple and is not bloated. It provides a set of fixed tasks that is not meant to be modified. If you are developing and require the flexibility of modifying the task and the details that come along with it, we recommend copying and writing your own custom `scoring_function` in your main script instead of importing from `tasks`.

The `tasks` directory also serves as a way to maintain a QD benchmark task suite that can be easily accesed. We implement several benchmark task across a range of domains. The tasks here are classical tasks from QD literature as well as more recent benchmarks tasks proposed at the [QD Benchmarks Workshop at GECCO 2022](https://quality-diversity.github.io/workshop).

## Arm
| Task           | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|----------------|----------------------|------------------|-----------------------|-------------------|-------------|
| Arm            | n                    | $[0,1]^n$        | 2                     | $[0,1]^2$         |             |
| Stochastic Arm | n                    | $[0,1]^n$        | 2                     | $[0,1]^2$         |             |

Notes:
- the parameter space is normalized between $[0,1]$ which corresponds to $[0,2\pi]$
- the descriptor space (end-effector x-y position) is normalized between $[0,1]$

## Standard Functions
| Task                 | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|----------------------|----------------------|------------------|-----------------------|-------------------|-------------|
| Sphere               | n                    | $[0,1]^n$        | 2                     | $[0,1]^n$         |             |
| Rastrigin            | n                    | $[0,1]^n$        | 2                     | $[0,1]^n$         |             |
| Rastrigin-Projection | n                    | $[0,1]^n$        | 2                     | $[0,1]^n$         |             |

## Hyper-Volume Functions
"Hypervolume-based Benchmark Functions for Quality Diversity Algorithms" by Jean-Baptiste Mouret

| Task                  | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|-----------------------|----------------------|------------------|-----------------------|-------------------|-------------|
| Square                | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Checkered             | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Empty Circle          | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Non-continous Islands | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |
| Continous Islands     | n                    | $[0,1]^n$        | n                     | $[0,1]^n$         |             |

## QD Suite
"Towards QD-suite: developing a set of benchmarks for Quality-Diversity algorithms" by Achkan Salehi and Stephane Doncieux

| Task                           | Parameter Dimensions | Parameter Bounds                                                               | Descriptor Dimensions                 | Descriptor Bounds                                                           | Description |
|--------------------------------|----------------------|--------------------------------------------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------|-------------|
| archimedean-spiral-v0          | 1                    | $[0,\alpha\pi]^n$ (angle param.)<br/> $[0,max-arc-length]$ (arc length param.) | 1 (geodesic BD)<br/> 2 (euclidean BD) | $[0,max-arc-length]$ (geodesic BD)<br/> $[-radius,radius]^2$ (euclidean BD) |             |
| SSF-v0                         | $n$                  | Unbounded                                                                      | 1                                     | $[ 0 ,$ ∞ $)$                                                               |             |
| deceptive-evolvability-v0<br/> | $n$ (2 by default)   | Bounded area including the two gaussian peaks                                  | 1                                     | $[0,max-sum-gaussians]$                                                     |             |

```python
import math
from qdax.tasks.qd_suite import archimedean_spiral_v0_angle_euclidean_task

task = archimedean_spiral_v0_angle_euclidean_task

# Get scoring function
scoring_fn = task.scoring_function

# Get initial batch of parameters
initial_params = task.get_initial_parameters(batch_size=...)

# Get Task Properties (parameter space, descriptor space, and grid_shape)
min_param, max_param = task.get_min_max_params()
min_desc, max_desc = task.get_bounded_min_max_descriptor()  # To consider bounded Descriptor space
# If the task has a descriptor space that is not bounded, then the unbounded descriptor
# space can be obtained via the following:
# min_bd, max_bd = task.get_min_max_bd()

bd_size = task.get_bd_size()
if bd_size == 1:
    grid_shape = (100,)
elif bd_size == 2:
    grid_shape = (100, 100)
else:
    resolution_per_axis = math.floor(math.pow(10000., 1. / bd_size))
    grid_shape = tuple([resolution_per_axis for _ in range(bd_size)])
```

## Brax-RL
| Task            | Parameter Dimensions | Parameter Bounds | Descriptor Dimensions | Descriptor Bounds | Description |
|-----------------|----------------------|------------------|-----------------------|-------------------|-------------|
| pointmaze       | NN params            | Unbounded        | 2                     |                   |             |
| hopper_uni      | NN params            | Unbounded        | 1                     | $[0,1]$           |             |
| walker2d_uni    | NN params            | Unbounded        | 2                     | $[0,1]^2$         |             |
| halfcheetah_uni | NN params            | Unbounded        | 2                     | $[0,1]^2$         |             |
| ant_uni         | NN params            | Unbounded        | 4                     | $[0,1]^4$         |             |
| humanoid_uni    | NN params            | Unbounded        | 2                     | $[0,1]^2$         |             |
| ant_omni        | NN params            | Unbounded        | 2                     | $[-30,30]^2$      |             |
| humanoid_omni   | NN params            | Unbounded        | 2                     | $[-30,30]^2$      |             |
| anttrap         | NN params            | Unbounded        | 2                     |                   |             |
| antmaze         | NN params            | Unbounded        | 2                     |                   |             |

Notes:
- the parameter dimensions for default Brax-RL tasks depend on the size and architecture of the neural network used and can be customized and changed easily. If not set, a network size of two hidden layers of size 64 is used.
