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
from qdax.environments.bd_extractors import AuroraExtraInfoNormalization
from qdax.types import Params, RNGKey

Array = Any
PRNGKey = Any


def get_model(
    obs_size: int, teacher_force: bool = False, hidden_size: int = 10
) -> Seq2seq:
    """
    Returns a seq2seq model.

    Args:
        obs_size: the size of the observation.
        teacher_force: whether to use teacher forcing.
        hidden_size: the size of the hidden layer (i.e. the encoding).
    """
    return Seq2seq(
        teacher_force=teacher_force, hidden_size=hidden_size, obs_size=obs_size
    )


def get_initial_params(
    model: Seq2seq, random_key: PRNGKey, encoder_input_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    """
    Returns the initial parameters of a seq2seq model.

    Args:
        model: the seq2seq model.
        random_key: the random number generator.
        encoder_input_shape: the shape of the encoder input.
    """
    random_key, rng1, rng2, rng3 = jax.random.split(random_key, 4)
    variables = model.init(
        {"params": rng1, "lstm": rng2, "dropout": rng3},
        jnp.ones(encoder_input_shape, jnp.float32),
        jnp.ones(encoder_input_shape, jnp.float32),
    )
    return variables["params"]  # type: ignore


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: Array,
    lstm_random_key: PRNGKey,
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Trains for one step.

    Args:
        state: the training state.
        batch: the batch of data.
        lstm_random_key: the random number key.
    """

    """Trains one step."""
    lstm_key = jax.random.fold_in(lstm_random_key, state.step)
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
    random_key: RNGKey,
    repertoire: UnstructuredRepertoire,
    params: Params,
    epoch: int,
    model: Seq2seq,
    batch_size: int = 128,
) -> AuroraExtraInfoNormalization:

    if epoch > 100:
        num_epochs = 25
        alpha = 0.0001  # Gradient step size
    else:
        num_epochs = 100
        alpha = 0.0001

    # compute mean/std of the obs for normalization
    mean_obs = jnp.nanmean(repertoire.observations, axis=(0, 1))
    std_obs = jnp.nanstd(repertoire.observations, axis=(0, 1))
    # the std where they were NaNs was set to zero. But here we divide by the
    # std, so we replace the zeros by inf here.
    std_obs = jnp.where(std_obs == 0, x=jnp.inf, y=std_obs)

    # TODO: maybe we could just compute this data on the valid dataset

    # create optimizer and optimized state
    tx = optax.adam(alpha)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    ###########################################################################
    # Shuffling indexes of valid individuals in the repertoire
    ###########################################################################

    # size of the repertoire
    repertoire_size = repertoire.max_size

    # number of individuals in the repertoire
    num_indivs = repertoire.get_number_genotypes()

    # select repertoire_size indexes going from 0 to the total number of
    # valid individuals. Those indexes will be used to select the individuals
    # in the training dataset.
    random_key, key_select_p1 = jax.random.split(random_key, 2)
    idx_p1 = jax.random.randint(
        key_select_p1, shape=(repertoire_size,), minval=0, maxval=num_indivs
    )

    # get indexes where fitness is not -inf. Those are the valid individuals.
    indexes = jnp.argwhere(
        jnp.logical_not(jnp.isinf(repertoire.fitnesses)), size=repertoire_size
    )
    indexes = jnp.transpose(indexes, axes=(1, 0))

    # get corresponding indices for the flattened repertoire fitnesses
    indiv_indices = jnp.array(
        jnp.ravel_multi_index(indexes, repertoire.fitnesses.shape, mode="clip")
    ).astype(int)

    # filter those indices to get only the indices of valid individuals
    valid_indexes = indiv_indices.at[idx_p1].get()

    # Normalising Dataset
    steps_per_epoch = repertoire.observations.shape[0] // batch_size

    loss_val = 0.0
    for epoch in range(num_epochs):
        random_key, shuffle_key = jax.random.split(random_key, 2)
        valid_indexes = jax.random.permutation(shuffle_key, valid_indexes, axis=0)

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

            state, loss_val = train_step(state, batch, random_key)

        # To see the actual value we cannot jit this function (i.e. the _one_es_epoch
        # function nor the train function)
        print("Eval epoch: {}, loss: {:.4f}".format(epoch + 1, loss_val))

    params = state.params

    return AuroraExtraInfoNormalization.create(params, mean_obs, std_obs)
