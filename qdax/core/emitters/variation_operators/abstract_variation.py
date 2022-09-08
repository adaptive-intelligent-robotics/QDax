import abc
from typing import Optional, Tuple

import jax
from chex import ArrayTree
from jax import numpy as jnp

from qdax.core.emitters.emitter import EmitterState
from qdax.types import Genotype, RNGKey


class VariationOperator(metaclass=abc.ABCMeta):
    def __init__(self, minval: Optional[float] = None, maxval: Optional[float] = None):
        if minval is not None and maxval is not None:
            assert minval < maxval, "minval must be smaller than maxval"
        self._minval = minval
        self._maxval = maxval

    @property
    @abc.abstractmethod
    def number_parents_to_select(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def number_genotypes_returned(self) -> int:
        ...

    def calculate_number_parents_to_select(self, batch_size: int) -> int:
        assert batch_size % self.number_genotypes_returned == 0, (
            "The batch size should be a multiple of the "
            "number of genotypes returned after each variation"
        )
        return (
            self.number_parents_to_select * batch_size // self.number_genotypes_returned
        )

    @abc.abstractmethod
    def apply_without_clip(
        self, genotypes: Genotype, emitter_state: EmitterState, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        ...

    def _clip(self, gen: Genotype) -> Genotype:
        if (self._minval is not None) or (self._maxval is not None):
            gen = jax.tree_map(
                lambda _gen: jnp.clip(_gen, self._minval, self._maxval), gen
            )
        return gen

    def apply_with_clip(
        self,
        genotypes: Genotype,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        new_genotypes, random_key = self.apply_without_clip(
            genotypes, emitter_state, random_key
        )
        new_genotypes = self._clip(new_genotypes)
        return new_genotypes, random_key

    def _divide_genotypes(
        self,
        genotypes: Genotype,
    ) -> Tuple[Genotype, ...]:
        tuple_genotypes = tuple(
            jax.tree_map(
                lambda x: x[index_start :: self.number_parents_to_select], genotypes
            )
            for index_start in range(self.number_parents_to_select)
        )
        return tuple_genotypes

    @staticmethod
    def get_tree_keys(
        genotype: Genotype, random_key: RNGKey
    ) -> Tuple[ArrayTree, RNGKey]:
        nb_leaves = len(jax.tree_leaves(genotype))
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=nb_leaves)
        keys_tree = jax.tree_unflatten(jax.tree_structure(genotype), subkeys)
        return keys_tree, random_key

    @staticmethod
    def _get_array_keys_for_each_gen(key: RNGKey, gen_tree: Genotype) -> jnp.ndarray:
        subkeys = jax.random.split(key, num=gen_tree.shape[0])
        return jnp.asarray(subkeys)

    @staticmethod
    def get_keys_arrays_tree(
        gen_tree: Genotype, random_key: RNGKey
    ) -> Tuple[ArrayTree, RNGKey]:
        keys_tree, random_key = VariationOperator.get_tree_keys(gen_tree, random_key)
        keys_arrays_tree = jax.tree_map(
            VariationOperator._get_array_keys_for_each_gen, keys_tree, gen_tree
        )
        return keys_arrays_tree, random_key

    @staticmethod
    def _get_random_positions_to_change(
        genotypes_tree: Genotype,
        variation_rate: float,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        def _get_indexes_positions_cross_over(
            _gen: Genotype, _key: RNGKey
        ) -> jnp.ndarray:
            num_positions = _gen.shape[0]
            positions = jnp.arange(start=0, stop=num_positions)
            num_positions_to_change = int(variation_rate * num_positions)
            _key, subkey = jax.random.split(_key)
            selected_positions = jax.random.choice(
                key=subkey, a=positions, shape=(num_positions_to_change,), replace=False
            )
            return selected_positions

        random_key, _subkey = jax.random.split(random_key)

        keys_arrays_tree, random_key = VariationOperator.get_keys_arrays_tree(
            genotypes_tree, random_key
        )

        return (
            jax.tree_map(
                jax.vmap(_get_indexes_positions_cross_over),
                genotypes_tree,
                keys_arrays_tree,
            ),
            random_key,
        )

    @staticmethod
    def _get_sub_genotypes(
        genotypes_tree: Genotype,
        selected_positions: jnp.ndarray,
    ) -> Genotype:
        return jax.tree_map(
            jax.vmap(lambda _x, _i: _x[_i]), genotypes_tree, selected_positions
        )

    @staticmethod
    def _set_sub_genotypes(
        genotypes_tree: Genotype,
        selected_positions: jnp.ndarray,
        new_genotypes: Genotype,
    ) -> Genotype:
        return jax.tree_map(
            jax.vmap(lambda _x, _i, _y: _x.at[_i].set(_y)),
            genotypes_tree,
            selected_positions,
            new_genotypes,
        )
