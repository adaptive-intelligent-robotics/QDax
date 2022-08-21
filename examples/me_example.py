import functools

import jax
import matplotlib.pyplot as plt

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.arm import arm_scoring_function
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import plot_2d_map_elites_repertoire

seed = 42
num_param_dimensions = 8  # num DoF arm
init_batch_size = 100
batch_size = 2048
num_iterations = 1000
grid_shape = (100, 100)
min_param = 0.0
max_param = 1.0
min_bd = 0.0
max_bd = 1.0

# Init a random key
random_key = jax.random.PRNGKey(seed)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
init_variables = jax.random.uniform(
    subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Define emitter
variation_fn = functools.partial(
    isoline_variation,
    iso_sigma=0.005,
    line_sigma=0,
    minval=min_param,
    maxval=max_param,
)
mixing_emitter = MixingEmitter(
    mutation_fn=lambda x, y: (x, y),
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=batch_size,
)

# Define a metrics function
metrics_fn = functools.partial(
    default_qd_metrics,
    qd_offset=0.0,
)

# Instantiate MAP-Elites
map_elites = MAPElites(
    scoring_function=arm_scoring_function,
    emitter=mixing_emitter,
    metrics_function=metrics_fn,
)

# Compute the centroids
centroids = compute_euclidean_centroids(
    grid_shape=grid_shape,
    minval=min_bd,
    maxval=max_bd,
)

# Initializes repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init(
    init_variables, centroids, random_key
)

# Run MAP-Elites loop
for _ in range(num_iterations):
    (repertoire, emitter_state, metrics, random_key,) = map_elites.update(
        repertoire,
        emitter_state,
        random_key,
    )

# plot archive
fig, axes = plot_2d_map_elites_repertoire(
    centroids=repertoire.centroids,
    repertoire_fitnesses=repertoire.fitnesses,
    minval=min_bd,
    maxval=max_bd,
    repertoire_descriptors=repertoire.descriptors,
    vmin=-0.2,
    vmax=0.0,
)
plt.show()
