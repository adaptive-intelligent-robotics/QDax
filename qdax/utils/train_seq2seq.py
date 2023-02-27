# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""seq2seq addition example."""

# See issue #620.
# pytype: disable=wrong-keyword-args

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import flags
from flax.training import train_state

from qdax.utils.seq2seq_model import Seq2seq

Array = Any
FLAGS = flags.FLAGS
PRNGKey = Any

flags.DEFINE_string("workdir", default=".", help="Where to store log output.")

flags.DEFINE_float(
    "learning_rate", default=0.003, help=("The learning rate for the Adam optimizer.")
)

flags.DEFINE_integer("batch_size", default=128, help=("Batch size for training."))

flags.DEFINE_integer("hidden_size", default=16, help=("Hidden size of the LSTM."))

flags.DEFINE_integer("num_train_steps", default=10000, help=("Number of train steps."))

flags.DEFINE_integer(
    "decode_frequency",
    default=200,
    help=("Frequency of decoding during training, e.g. every 1000 steps."),
)

flags.DEFINE_integer(
    "max_len_query_digit", default=3, help=("Maximum length of a single input digit.")
)


def get_model(obs_size, teacher_force: bool = False, hidden_size=10) -> Seq2seq:
    return Seq2seq(
        teacher_force=teacher_force, hidden_size=hidden_size, obs_size=obs_size
    )


def get_initial_params(
    model: Seq2seq, rng: PRNGKey, encoder_input_shape
) -> Dict[str, Any]:
    """Returns the initial parameters of a seq2seq model."""
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    variables = model.init(
        {"params": rng1, "lstm": rng2, "dropout": rng3},
        jnp.ones(encoder_input_shape, jnp.float32),
        jnp.ones(encoder_input_shape, jnp.float32),
    )
    return variables["params"]


@jax.jit
def train_step(
    state: train_state.TrainState, batch: Array, lstm_rng: PRNGKey
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Trains one step."""
    lstm_key = jax.random.fold_in(lstm_rng, state.step)
    dropout_key, lstm_key = jax.random.split(lstm_key, 2)
    # Shift Input by One to avoid leakage
    batch_decoder = jnp.roll(batch, shift=1, axis=1)
    ### Large number as zero token
    batch_decoder = batch_decoder.at[:, 0, :].set(-1000)

    def loss_fn(params):
        logits, _ = state.apply_fn(
            {"params": params},
            batch,
            batch_decoder,
            rngs={"lstm": lstm_key, "dropout": dropout_key},
        )

        def squared_error(x, y):
            return jnp.inner(y - x, y - x) / 2.0

        def mean_squared_error(x, y):
            return jnp.inner(y - x, y - x) / x.shape[-1]

        # res = jax.vmap(squared_error)(logits, batch)
        # res = jax.vmap(squared_error)(jnp.reshape(logits,(logits.shape[0],-1)),jnp.reshape(batch,(batch.shape[0],-1)))
        res = jax.vmap(mean_squared_error)(
            jnp.reshape(logits.at[:, :-1, ...].get(), (logits.shape[0], -1)),
            jnp.reshape(
                batch_decoder.at[:, 1:, ...].get(), (batch_decoder.shape[0], -1)
            ),
        )
        loss = jnp.mean(res, axis=0)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss_val


def lstm_ae_train(key, repertoire, params, epoch, hidden_size=10):
    batch_size = 128  # 2048

    if epoch > 100:
        num_epochs = 25
        alpha = 0.0001  # Gradient step size
    else:
        num_epochs = 100
        alpha = 0.0001  # Gradient step size

    rng, key, key_selection = jax.random.split(key, 3)
    dimensions_data = jnp.prod(jnp.asarray(repertoire.observations.shape[1:]))

    # get the model used (seq2seq)
    model = get_model(
        repertoire.observations.shape[-1], teacher_force=True, hidden_size=hidden_size
    )

    print("Beginning of the lstm ae training: ")
    print("Repertoire observation: ", repertoire.observations)

    print("Repertoire fitnesses: ", repertoire.fitnesses)

    # compute mean/std of the obs for normalization
    mean_obs = jnp.nanmean(repertoire.observations, axis=(0, 1))
    std_obs = jnp.nanstd(repertoire.observations, axis=(0, 1))

    print("Mean obs - wo NaN: ", mean_obs)
    print("Std obs - wo NaN: ", std_obs)

    # TODO: maybe we could just compute this data on the valid dataset

    # create optimizer and optimized state
    tx = optax.adam(alpha)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # size of the repertoire
    repertoire_size = repertoire.centroids.shape[0]
    print("Repertoire size: ", repertoire_size)

    # number of individuals in the repertoire
    num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)
    print("Number of individuals: ", num_indivs)

    # select repertoire_size indexes going from 0 to num_indivs
    # TODO: WHY??
    key_select_p1, rng = jax.random.split(key_selection, 2)
    idx_p1 = jax.random.randint(
        key_select_p1, shape=(repertoire_size,), minval=0, maxval=num_indivs
    )
    print("idx p1: ", idx_p1)

    # TODO: what is the diff with repertoire_size??
    tot_indivs = repertoire.fitnesses.ravel().shape[0]
    print("Total individuals: ", tot_indivs)

    # get indexes where fitness is not -inf??
    indexes = jnp.argwhere(
        jnp.logical_not(jnp.isinf(repertoire.fitnesses)), size=tot_indivs
    )
    indexes = jnp.transpose(indexes, axes=(1, 0))
    print("Indexes: ", indexes)

    # ???
    indiv_indices = jnp.array(
        jnp.ravel_multi_index(indexes, repertoire.fitnesses.shape, mode="clip")
    ).astype(int)
    print("Indiv indices: ", indexes)

    # ???
    valid_indexes = indiv_indices.at[idx_p1].get()
    print("Valid indexes: ", valid_indexes)

    # Normalising Dataset
    # training_dataset = (repertoire.observations.at[valid_indexes].get()-mean_obs)/std_obs #jnp.where(std_obs==0,mean_obs,std_obs)
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
        ) / std_obs  # jnp.where(std_obs==0,mean_obs,std_obs)
        training_dataset = training_dataset.at[valid_indexes].get()

        if epoch == 0:
            print("Training dataset for first epoch: ", training_dataset)
            print("Training dataset first data for first epoch: ", training_dataset[0])

        for i in range(steps_per_epoch):
            batch = jnp.asarray(
                training_dataset.at[
                    (i * batch_size) : (i * batch_size) + batch_size, :, :
                ].get()
            )
            # print(batch)
            if batch.shape[0] < batch_size:
                # print(batch.shape)
                continue
            state, loss_val = train_step(state, batch, rng)

        ### To see the actual value we cannot jit this function (i.e. the _one_es_epoch function nor the train function)
        print("Eval epoch: {}, loss: {:.4f}".format(epoch + 1, loss_val))

        # TODO: put this in metrics so we can jit the function and see the metrics
        # TODO: not urgent because the training is not that long

    # return repertoire.replace(ae_params=state.params,mean_obs=mean_obs,std_obs=std_obs)

    train_step.clear_cache()
    del tx
    del model
    params = state.params
    del state

    return params, mean_obs, std_obs
