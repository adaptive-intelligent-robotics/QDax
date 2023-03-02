"""seq2seq addition example

Inspired by Flax library -
https://github.com/google/flax/blob/main/examples/seq2seq/train.py

Copyright 2022 The Flax Authors.
Licensed under the Apache License, Version 2.0 (the "License")
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.neuroevolution.networks.seq2seq_networks import Seq2seq
from qdax.types import Observation, Params, RNGKey

Array = Any
PRNGKey = Any


def get_model(
    obs_size: int, teacher_force: bool = False, hidden_size: int = 10
) -> Seq2seq:
    return Seq2seq(
        teacher_force=teacher_force, hidden_size=hidden_size, obs_size=obs_size
    )


def get_initial_params(
    model: Seq2seq, rng: PRNGKey, encoder_input_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    """Returns the initial parameters of a seq2seq model."""
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    variables = model.init(
        {"params": rng1, "lstm": rng2, "dropout": rng3},
        jnp.ones(encoder_input_shape, jnp.float32),
        jnp.ones(encoder_input_shape, jnp.float32),
    )
    return variables["params"]  # type: ignore


@jax.jit
def train_step(
    state: train_state.TrainState, batch: Array, lstm_rng: PRNGKey
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Trains one step."""
    lstm_key = jax.random.fold_in(lstm_rng, state.step)
    dropout_key, lstm_key = jax.random.split(lstm_key, 2)

    # Shift input by one to avoid leakage
    batch_decoder = jnp.roll(batch, shift=1, axis=1)

    # Large number as zero token
    batch_decoder = batch_decoder.at[:, 0, :].set(-1000)

    def loss_fn(params: Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits, _ = state.apply_fn(
            {"params": params},
            batch,
            batch_decoder,
            rngs={"lstm": lstm_key, "dropout": dropout_key},
        )

        def mean_squared_error(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.inner(y - x, y - x) / x.shape[-1]

        res = jax.vmap(mean_squared_error)(
            jnp.reshape(logits.at[:, :-1, ...].get(), (logits.shape[0], -1)),
            jnp.reshape(
                batch_decoder.at[:, 1:, ...].get(), (batch_decoder.shape[0], -1)
            ),
        )
        loss = jnp.mean(res, axis=0)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, _logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss_val


def lstm_ae_train(
    key: RNGKey,
    repertoire: UnstructuredRepertoire,
    params: Params,
    epoch: int,
    hidden_size: int = 10,
    batch_size: int = 128,
) -> Tuple[Params, Observation, Observation]:
    if epoch > 100:
        num_epochs = 25

        # Gradient step size
        alpha = 0.0001
    else:
        num_epochs = 100

        # Gradient step size
        alpha = 0.0001

    rng, key, key_selection = jax.random.split(key, 3)

    # get the model used (seq2seq)
    model = get_model(
        repertoire.observations.shape[-1], teacher_force=True, hidden_size=hidden_size
    )

    # compute mean/std of the obs for normalization
    mean_obs = jnp.nanmean(repertoire.observations, axis=(0, 1))
    std_obs = jnp.nanstd(repertoire.observations, axis=(0, 1))

    # TODO: maybe we could just compute this data on the valid dataset

    # create optimizer and optimized state
    tx = optax.adam(alpha)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # size of the repertoire
    repertoire_size = repertoire.centroids.shape[0]

    # number of individuals in the repertoire
    num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

    # select repertoire_size indexes going from 0 to num_indivs
    # TODO: WHY??
    key_select_p1, rng = jax.random.split(key_selection, 2)
    idx_p1 = jax.random.randint(
        key_select_p1, shape=(repertoire_size,), minval=0, maxval=num_indivs
    )

    # TODO: what is the diff with repertoire_size??
    tot_indivs = repertoire.fitnesses.ravel().shape[0]

    # get indexes where fitness is not -inf??
    indexes = jnp.argwhere(
        jnp.logical_not(jnp.isinf(repertoire.fitnesses)), size=tot_indivs
    )
    indexes = jnp.transpose(indexes, axes=(1, 0))

    # ???
    indiv_indices = jnp.array(
        jnp.ravel_multi_index(indexes, repertoire.fitnesses.shape, mode="clip")
    ).astype(int)

    # ???
    valid_indexes = indiv_indices.at[idx_p1].get()

    # Normalising Dataset
    steps_per_epoch = repertoire.observations.shape[0] // batch_size

    loss_val = 0.0
    for epoch in range(num_epochs):
        rng, shuffle_key = jax.random.split(rng, 2)
        valid_indexes = jax.random.permutation(shuffle_key, valid_indexes, axis=0)

        # TODO: the std where they were NaNs is set to zero. But here we divide by the
        # std, so NaNs appear here...
        # std_obs += 1e-6

        std_obs = jnp.where(std_obs == 0, x=jnp.inf, y=std_obs)

        # create dataset with the observation from the sample of valid indexes
        training_dataset = (
            repertoire.observations.at[valid_indexes, ...].get() - mean_obs
        ) / std_obs
        training_dataset = training_dataset.at[valid_indexes].get()

        for i in range(steps_per_epoch):
            batch = jnp.asarray(
                training_dataset.at[
                    (i * batch_size) : (i * batch_size) + batch_size, :, :
                ].get()
            )

            if batch.shape[0] < batch_size:
                # print(batch.shape)
                continue

            state, loss_val = train_step(state, batch, rng)

        # To see the actual value we cannot jit this function (i.e. the _one_es_epoch
        # function nor the train function)
        print("Eval epoch: {}, loss: {:.4f}".format(epoch + 1, loss_val))

        # TODO: put this in metrics so we can jit the function and see the metrics
        # TODO: not urgent because the training is not that long

    params = state.params

    return params, mean_obs, std_obs
