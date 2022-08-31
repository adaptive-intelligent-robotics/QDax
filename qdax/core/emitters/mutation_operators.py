"""File defining mutation and crossover functions."""
import abc
import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import ArrayTree
from matplotlib import pyplot as plt

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
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


class ComposerVariations(VariationOperator):
    def __init__(
        self,
        variations_operators_list: List[VariationOperator],
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(minval, maxval)
        self.variations_list = variations_operators_list

    @property
    def number_parents_to_select(self) -> int:
        numbers_to_select = map(
            lambda x: x.number_parents_to_select, self.variations_list
        )
        return math.prod(numbers_to_select)

    @property
    def number_genotypes_returned(self) -> int:
        numbers_to_return = map(
            lambda x: x.number_genotypes_returned, self.variations_list
        )
        return math.prod(numbers_to_return)

    def apply_without_clip(
        self, genotypes: Genotype, emitter_state: EmitterState, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        for variation in self.variations_list:
            genotypes, random_key = variation.apply_with_clip(
                genotypes, emitter_state, random_key
            )
        return genotypes, random_key


class Mutation(VariationOperator, abc.ABC):
    def __init__(
        self,
        mutation_rate: float,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(minval, maxval)
        self.mutation_rate = mutation_rate

    @property
    def number_parents_to_select(self) -> int:
        return 1

    @property
    def number_genotypes_returned(self) -> int:
        return 1

    def apply_without_clip(
        self,
        genotypes: Genotype,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        selected_indices, random_key = self._get_random_positions_to_change(
            genotypes, self.mutation_rate, random_key
        )
        selected_gens = self._get_sub_genotypes(
            genotypes, selected_positions=selected_indices
        )
        selected_gens_mutated, random_key = self._mutate(selected_gens, random_key)
        new_genotypes = self._set_sub_genotypes(
            genotypes, selected_indices, selected_gens_mutated
        )
        return new_genotypes, random_key

    @abc.abstractmethod
    def _mutate(self, gen: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        ...


class NormalMutation(Mutation):
    def __init__(
        self,
        sigma: float,
        mutation_rate: float = 1.0,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(mutation_rate=mutation_rate, minval=minval, maxval=maxval)
        self.sigma = sigma

    def _mutate(self, gen: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        array_keys, random_key = self.get_tree_keys(gen, random_key)

        def _variation_fn(_gen: Genotype, _key: RNGKey) -> Genotype:
            return _gen + jax.random.normal(key=_key, shape=_gen.shape) * self.sigma

        return jax.tree_map(_variation_fn, gen, array_keys), random_key


class PolynomialMutation(Mutation):
    def __init__(
        self,
        eta: float,
        minval: float,
        maxval: float,
        mutation_rate: float = 1.0,
    ):
        # for polynomial mutation, minval and maxval must be specified and finite
        super().__init__(mutation_rate=mutation_rate, minval=minval, maxval=maxval)
        self._eta = eta

    def _mutate(self, gen: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        def _mutate_single_subgen_array(
            _subgen_array: jnp.ndarray, _random_key: RNGKey
        ) -> jnp.ndarray:
            assert self._minval is not None and self._maxval is not None

            delta_1 = (_subgen_array - self._minval) / (self._maxval - self._minval)
            delta_2 = (self._maxval - _subgen_array) / (self._maxval - self._minval)
            mutpow = 1.0 / (1.0 + self._eta)

            # Randomly select where to put delta_1 and delta_2
            _random_key, subkey = jax.random.split(_random_key)
            rand = jax.random.uniform(
                key=subkey,
                shape=delta_1.shape,
                minval=0,
                maxval=1,
                dtype=jnp.float32,
            )

            value1 = 2.0 * rand + (
                jnp.power(delta_1, 1.0 + self._eta) * (1.0 - 2.0 * rand)
            )
            value2 = 2.0 * (1 - rand) + 2.0 * (
                jnp.power(delta_2, 1.0 + self._eta) * (rand - 0.5)
            )
            value1 = jnp.power(value1, mutpow) - 1.0
            value2 = 1.0 - jnp.power(value2, mutpow)

            delta_q = jnp.zeros_like(_subgen_array)
            delta_q = jnp.where(rand < 0.5, value1, delta_q)
            delta_q = jnp.where(rand >= 0.5, value2, delta_q)

            # Mutate values
            new_subgen_array = _subgen_array + delta_q * (self._maxval - self._minval)
            return new_subgen_array

        keys_arrays_tree, random_key = self.get_keys_arrays_tree(gen, random_key)

        new_gen = jax.tree_map(_mutate_single_subgen_array, gen, keys_arrays_tree)

        return new_gen, random_key


class CrossOver(VariationOperator, abc.ABC):
    def __init__(
        self,
        cross_over_rate: float = 1.0,
        returns_single_genotype: bool = True,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(minval, maxval)
        self.cross_over_rate = cross_over_rate
        self.returns_single_genotype = returns_single_genotype

    @property
    def number_parents_to_select(self) -> int:
        return 2

    @property
    def number_genotypes_returned(self) -> int:
        if self.returns_single_genotype:
            return 1
        else:
            return 2

    def apply_without_clip(
        self,
        genotypes: Genotype,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        gen_1, gen_2 = self._divide_genotypes(genotypes)
        selected_indices, random_key = self._get_random_positions_to_change(
            gen_1, self.cross_over_rate, random_key
        )
        subgen_1 = self._get_sub_genotypes(gen_1, selected_positions=selected_indices)
        subgen_2 = self._get_sub_genotypes(gen_2, selected_positions=selected_indices)

        if self.returns_single_genotype:

            new_subgen, random_key = self._cross_over(subgen_1, subgen_2, random_key)
            new_gen = self._set_sub_genotypes(gen_1, selected_indices, new_subgen)
            return new_gen, random_key
        else:
            # Not changing random key here to keep same noise for gen_tilde_1 and
            # gen_tilde_2 (as done in the literature)
            new_subgen_1, _ = self._cross_over(subgen_1, subgen_2, random_key)
            new_subgen_2, random_key = self._cross_over(subgen_2, subgen_1, random_key)

            new_gen_1 = self._set_sub_genotypes(gen_1, selected_indices, new_subgen_1)
            new_gen_2 = self._set_sub_genotypes(gen_2, selected_indices, new_subgen_2)

            new_gen = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                new_gen_1,
                new_gen_2,
            )
            return new_gen, random_key

    @abc.abstractmethod
    def _cross_over(
        self, gen_1: Genotype, gen_2: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        ...


class RecombinationCrossOver(CrossOver):
    def _cross_over(
        self, gen_original: Genotype, gen_exchange: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        # The exchange cross over is a simple exchange of the two genotypes
        # the proportion of the two genotypes that are changed is the
        # same as the cross-over rate the parts which are exchanged are
        # randomly selected in CrossOver
        return gen_exchange, random_key


class SBXCrossOver(CrossOver):
    def __init__(
        self,
        eta: float,
        minval: float,
        maxval: float,
        cross_over_rate: float = 1.0,
        returns_single_genotype: bool = True,
    ):
        super().__init__(cross_over_rate, returns_single_genotype, minval, maxval)
        self._eta = eta

    def _cross_over(
        self, gen_1: Genotype, gen_2: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        def _crossover_single_subgen_array(
            _subgen_array_1: jnp.ndarray,
            _subgen_array_2: jnp.ndarray,
            _random_key: RNGKey,
        ) -> jnp.ndarray:
            assert self._minval is not None and self._maxval is not None

            normalized_gen_1 = (_subgen_array_1 - self._minval) / (
                self._maxval - self._minval
            )
            normalized_gen_2 = (_subgen_array_2 - self._minval) / (
                self._maxval - self._minval
            )

            y1 = jnp.minimum(normalized_gen_1, normalized_gen_2)
            y2 = jnp.maximum(normalized_gen_1, normalized_gen_2)

            yl = 0.0
            yu = 1.0

            _random_key, _subkey = jax.random.split(_random_key)
            rand = jax.random.uniform(
                key=_random_key, shape=y1.shape, minval=0, maxval=1, dtype=jnp.float32
            )

            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta ** -(self._eta + 1.0)

            alpha_rand = rand * alpha

            betaq = jnp.where(
                rand <= 1.0 / alpha,
                (alpha_rand ** (1.0 / (self._eta + 1.0))),
                (1.0 / (2.0 - alpha_rand)) ** (1.0 / (self._eta + 1.0)),
            )

            c1 = 0.5 * (y1 + y2) - 0.5 * (y2 - y1) * betaq

            c1 = jnp.clip(c1, yl, yu)

            c1 = c1 * (self._maxval - self._minval) + self._minval

            return c1

        keys_arrays_tree, random_key = self.get_keys_arrays_tree(gen_1, random_key)
        new_gen = jax.tree_util.tree_map(
            jax.vmap(_crossover_single_subgen_array),
            gen_1,
            gen_2,
            keys_arrays_tree,
        )
        return new_gen, random_key


class IsolineVariationOperator(CrossOver):
    def __init__(
        self,
        iso_sigma: float,
        line_sigma: float,
        cross_over_rate: float = 1.0,
        returns_single_genotype: bool = True,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(
            cross_over_rate=cross_over_rate,
            returns_single_genotype=returns_single_genotype,
            minval=minval,
            maxval=maxval,
        )
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma

    def _cross_over(
        self, gen_1: Genotype, gen_2: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        # Computing line_noise
        random_key, key_line_noise = jax.random.split(random_key)
        batch_size = jax.tree_leaves(gen_1)[0].shape[0]
        line_noise = (
            jax.random.normal(key_line_noise, shape=(batch_size,)) * self._line_sigma
        )

        def _variation_fn(
            _x1: jnp.ndarray, _x2: jnp.ndarray, _random_key: RNGKey
        ) -> jnp.ndarray:
            iso_noise = (
                jax.random.normal(_random_key, shape=_x1.shape) * self._iso_sigma
            )
            x = (_x1 + iso_noise) + jax.vmap(jnp.multiply)((_x2 - _x1), line_noise)

            # Back in bounds if necessary (floating point issues)
            if (self._minval is not None) or (self._maxval is not None):
                x = jnp.clip(x, self._minval, self._maxval)
            return x

        # create a tree with random keys
        keys_tree, random_key = self.get_tree_keys(gen_1, random_key)

        # apply isolinedd to each branch of the tree
        gen_new = jax.tree_map(_variation_fn, gen_1, gen_2, keys_tree)

        return gen_new, random_key


class Selector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select(
        self,
        number_parents_to_select: int,
        repertoire: Repertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]:
        ...


class UniformSelector(Selector):
    def select(
        self,
        number_parents_to_select: int,
        repertoire: Repertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]:
        """
        Uniform selection of parents
        """
        selected_parents, random_key = repertoire.sample(
            random_key, number_parents_to_select
        )
        return selected_parents, emitter_state, random_key


class SelectionVariationEmitter(Emitter):
    def __init__(
        self,
        batch_size: int,
        variation_operator: VariationOperator,
        selector: Selector = None,
    ):
        self._batch_size = batch_size
        self._variation_operator = variation_operator

        if selector is not None:
            self._selector = selector
        else:
            self._selector = UniformSelector()

    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        number_parents_to_select = (
            self._variation_operator.calculate_number_parents_to_select(
                self._batch_size
            )
        )
        genotypes, emitter_state, random_key = self._selector.select(
            number_parents_to_select, repertoire, emitter_state, random_key
        )
        new_genotypes, random_key = self._variation_operator.apply_with_clip(
            genotypes, emitter_state, random_key
        )
        return new_genotypes, random_key


def _test() -> None:
    v1 = PolynomialMutation(
        eta=20,
        minval=0.0,
        maxval=1.0,
    )
    v2 = IsolineVariationOperator(
        iso_sigma=0.1,
        line_sigma=0.1,
        cross_over_rate=0.5,
        returns_single_genotype=False,
        minval=None,
        maxval=None,
    )
    v4 = SBXCrossOver(
        eta=20.0,
        minval=0.0,
        maxval=1.0,
    )
    v3 = RecombinationCrossOver(cross_over_rate=0.5, returns_single_genotype=False)
    v = ComposerVariations(
        variations_operators_list=[v1, v2, v3, v4], minval=None, maxval=None
    )
    gen_1 = {
        "a": jnp.concatenate(
            [0.3 * jnp.ones((1, 300)), 0.7 * jnp.ones((1, 300))], axis=0
        )
    }
    # gen_2 = {"a": jnp.array([[0., -1.], [0.5, 0.5]])}
    random_key = jax.random.PRNGKey(5)
    res = v.apply_with_clip(gen_1, None, random_key)[0]["a"]

    fig, ax = plt.subplots()
    ax.scatter(res[0, :], jnp.zeros_like(res[0, :]), s=1, linewidths=0)
    ax.set_xlim([0, 1])
    plt.show()


if __name__ == "__main__":
    _test()
