"""seq2seq example: Mode code.

Inspired by Flax library -
https://github.com/google/flax/blob/main/examples/seq2seq/models.py

Copyright 2022 The Flax Authors.
Licensed under the Apache License, Version 2.0 (the "License")
"""

import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

Array = Any
PRNGKey = Any


class EncoderLSTM(nn.Module):
    """EncoderLSTM Module wrapped in a lifted scan transform."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: Tuple[Array, Array], x: Array
    ) -> Tuple[Tuple[Array, Array], Array]:
        """Applies the module."""
        lstm_state, is_eos = carry
        features = lstm_state[0].shape[-1]
        new_lstm_state, y = nn.LSTMCell(features)(lstm_state, x)

        def select_carried_state(new_state: Array, old_state: Array) -> Array:
            return jnp.where(is_eos[:, np.newaxis], old_state, new_state)

        # LSTM state is a tuple (c, h).
        carried_lstm_state = tuple(
            select_carried_state(*s) for s in zip(new_lstm_state, lstm_state)
        )

        return (carried_lstm_state, is_eos), y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> Tuple[Array, Array]:
        # Use a dummy key since the default state init fn is just zeros.
        return nn.LSTMCell(hidden_size, parent=None).initialize_carry(  # type: ignore
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class Encoder(nn.Module):
    """LSTM encoder, returning state after finding the EOS token in the input."""

    hidden_size: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        batch_size = inputs.shape[0]
        lstm = EncoderLSTM(name="encoder_lstm")
        init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)

        # We use the `is_eos` array to determine whether the encoder should carry
        # over the last lstm state, or apply the LSTM cell on the previous state.
        init_is_eos = jnp.zeros(batch_size, dtype=bool)
        init_carry = (init_lstm_state, init_is_eos)
        (final_state, _), _ = lstm(init_carry, inputs)

        return final_state


class DecoderLSTM(nn.Module):
    """DecoderLSTM Module wrapped in a lifted scan transform.

    Attributes:
      teacher_force: See docstring on Seq2seq module.
      obs_size: Size of the observations.
    """

    teacher_force: bool
    obs_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False, "lstm": True},
    )
    @nn.compact
    def __call__(self, carry: Tuple[Array, Array], x: Array) -> Array:
        """Applies the DecoderLSTM model."""

        lstm_state, last_prediction = carry
        if not self.teacher_force:
            x = last_prediction

        features = lstm_state[0].shape[-1]
        new_lstm_state, y = nn.LSTMCell(features)(lstm_state, x)

        logits = nn.Dense(features=self.obs_size)(y)

        return (lstm_state, logits), (logits, logits)


class Decoder(nn.Module):
    """LSTM decoder.

    Attributes:
      init_state: [batch_size, hidden_size]
        Initial state of the decoder (i.e., the final state of the encoder).
      teacher_force: See docstring on Seq2seq module.
      obs_size: Size of the observations.
    """

    teacher_force: bool
    obs_size: int

    @nn.compact
    def __call__(self, inputs: Array, init_state: Any) -> Tuple[Array, Array]:
        """Applies the decoder model.

        Args:
          inputs: [batch_size, max_output_len-1, obs_size]
            Contains the inputs to the decoder at each time step (only used when not
            using teacher forcing). Since each token at position i is fed as input
            to the decoder at position i+1, the last token is not provided.

        Returns:
          Pair (logits, predictions), which are two arrays of respectively decoded
          logits and predictions (in one hot-encoding format).
        """
        lstm = DecoderLSTM(teacher_force=self.teacher_force, obs_size=self.obs_size)
        init_carry = (init_state, inputs[:, 0])
        _, (logits, predictions) = lstm(init_carry, inputs)
        return logits, predictions


class Seq2seq(nn.Module):
    """Sequence-to-sequence class using encoder/decoder architecture.

    Attributes:
      teacher_force: whether to use `decoder_inputs` as input to the decoder at
        every step. If False, only the first input (i.e., the "=" token) is used,
        followed by samples taken from the previous output logits.
      hidden_size: int, the number of hidden dimensions in the encoder and decoder
        LSTMs.
      obs_size: the size of the observations.
      eos_id: EOS id.
    """

    teacher_force: bool
    hidden_size: int
    obs_size: int

    def setup(self) -> None:
        self.encoder = Encoder(hidden_size=self.hidden_size)
        self.decoder = Decoder(teacher_force=self.teacher_force, obs_size=self.obs_size)

    @nn.compact
    def __call__(
        self, encoder_inputs: Array, decoder_inputs: Array
    ) -> Tuple[Array, Array]:
        """Applies the seq2seq model.

        Args:
          encoder_inputs: [batch_size, max_input_length, obs_size].
            padded batch of input sequences to encode.
          decoder_inputs: [batch_size, max_output_length, obs_size].
            padded batch of expected decoded sequences for teacher forcing.
            When sampling (i.e., `teacher_force = False`), only the first token is
            input into the decoder (which is the token "="), and samples are used
            for the following inputs. The second dimension of this tensor determines
            how many steps will be decoded, regardless of the value of
            `teacher_force`.

        Returns:
          Pair (logits, predictions), which are two arrays of length `batch_size`
          containing respectively decoded logits and predictions (in one hot
          encoding format).
        """
        # encode inputs
        init_decoder_state = self.encoder(encoder_inputs)

        # decode outputs
        logits, predictions = self.decoder(decoder_inputs, init_decoder_state)

        return logits, predictions

    def encode(self, encoder_inputs: Array) -> Array:
        # encode inputs
        init_decoder_state = self.encoder(encoder_inputs)
        final_output, _hidden_state = init_decoder_state
        return final_output
