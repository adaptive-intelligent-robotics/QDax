from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def vector_quantize(points, codebook):
    # ASSIGNMENT
    # codebook is centroids and is of shape -> (n_niches, bd_dim)
    # points is data collected to for the centroids to and is of shape -> (n_samples, bd_dim)
    # for each point - compute distance to of point to all the centroids/means
    # take the index (argmin) of the centroid/mean with the smallest distance to the point
    # Repeat this for every point in the sampled points (data)
    # end up with an array of indexes of the centroid for each point in the dataset (assignment) -> (n_samples)

    # distns an array of the distance between each of the points in the dataset and its assigned centroid/mean -> (n_samples)
    assignment = jax.vmap(
        lambda point: jnp.argmin(jax.vmap(jnp.linalg.norm)(codebook - point))
    )(points)
    distns = jax.vmap(jnp.linalg.norm)(codebook[assignment, :] - points)
    return assignment, distns


@partial(jax.jit, static_argnums=(2,))
def kmeans_run(key, points, k, thresh=1e-5):
    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = vector_quantize(points, prev_centroids)

        # Count number of points assigned per centroid
        # (Thanks to Jean-Baptiste Cordonnier for pointing this way out that is
        # much faster and let's be honest more readable!)
        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(a_min=1.0)  # clip to change 0/0 later to 0/1
        )

        # Sum over points in a centroid by zeroing others out
        new_centroids = (
            jnp.sum(
                jnp.where(
                    # axes: (data points, clusters, data dimension)
                    assignment[:, jnp.newaxis, jnp.newaxis]
                    == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
                    points[:, jnp.newaxis, :],
                    0.0,
                ),
                axis=0,
            )
            / counts
        )

        return new_centroids, jnp.mean(distortions), prev_distn

    # Run one iteration to initialize distortions and cause it'll never hurt...
    initial_indices = jax.random.shuffle(key, jnp.arange(points.shape[0]))[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None))
    # ...then iterate until convergence!
    centroids, distortion, _ = jax.lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion


@partial(jax.jit, static_argnums=(2, 3))
def kmeans(key, points, k, restarts, **kwargs):
    all_centroids, all_distortions = jax.vmap(
        lambda key: kmeans_run(key, points, k, **kwargs)
    )(jax.random.split(key, restarts))
    i = jnp.argmin(all_distortions)
    return all_centroids[i], all_distortions[i]


def cvt(n_niches, k1, k2):
    samples = 25000  # number of samples data points used to generate cetnroids (assumption here is tha BD is between - 0 and 1)
    dim = 2  # number of BD dimensions
    n_niches = n_niches
    x = jax.random.uniform(
        k1, shape=(samples, dim)
    )  # data samples in sampled unifromly based on bounds of BD space
    print("Data sameples shape: ", x.shape)

    centroids, distortions = kmeans(k2, x, n_niches, restarts=1)
    print("Centroids shape: ", centroids.shape)
    return centroids


def test_main():
    k1, k2 = jax.random.split(jax.random.PRNGKey(8), 2)
    centroids = cvt(10000, k1, k2)
