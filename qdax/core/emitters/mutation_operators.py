"""File defining mutation and crossover functions."""
import abc
import math
from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import ArrayTree

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Genotype, RNGKey


def _polynomial_mutation(
    x: jnp.ndarray,
    random_key: RNGKey,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
) -> jnp.ndarray:
    """Base polynomial mutation for one genotype.

    Proportion to mutate between 0 and 1
    Assumed to be of shape (genotype_dim,)

    Args:
        x: parameters.
        random_key: a random key
        proportion_to_mutate: the proportion of the given parameters
            that need to be mutated.
        eta: the inverse of the power of the mutation applied.
        minval: range of the perturbation applied by the mutation.
        maxval: range of the perturbation applied by the mutation.

    Returns:
        New parameters.
    """

    # Select positions to mutate
    num_positions = x.shape[0]
    positions = jnp.arange(start=0, stop=num_positions)
    num_positions_to_mutate = int(proportion_to_mutate * num_positions)
    random_key, subkey = jax.random.split(random_key)
    selected_positions = jax.random.choice(
        key=subkey, a=positions, shape=(num_positions_to_mutate,), replace=False
    )

    # New values
    mutable_x = x[selected_positions]
    delta_1 = (mutable_x - minval) / (maxval - minval)
    delta_2 = (maxval - mutable_x) / (maxval - minval)
    mutpow = 1.0 / (1.0 + eta)

    # Randomly select where to put delta_1 and delta_2
    random_key, subkey = jax.random.split(random_key)
    rand = jax.random.uniform(
        key=subkey,
        shape=delta_1.shape,
        minval=0,
        maxval=1,
        dtype=jnp.float32,
    )

    value1 = 2.0 * rand + (jnp.power(delta_1, 1.0 + eta) * (1.0 - 2.0 * rand))
    value2 = 2.0 * (1 - rand) + 2.0 * (jnp.power(delta_2, 1.0 + eta) * (rand - 0.5))
    value1 = jnp.power(value1, mutpow) - 1.0
    value2 = 1.0 - jnp.power(value2, mutpow)

    delta_q = jnp.zeros_like(mutable_x)
    delta_q = jnp.where(rand < 0.5, value1, delta_q)
    delta_q = jnp.where(rand >= 0.5, value2, delta_q)

    # Mutate values
    x = x.at[selected_positions].set(mutable_x + (delta_q * (maxval - minval)))

    # Back in bounds if necessary (floating point issues)
    x = jnp.clip(x, minval, maxval)

    return x


def polynomial_mutation(
    x: Genotype,
    random_key: RNGKey,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
) -> Tuple[Genotype, RNGKey]:
    """
    Polynomial mutation over several genotypes

    Parameters:
        x: array of genotypes to transform (real values only)
        random_key: RNG key for reproducibility.
            Assumed to be of shape (batch_size, genotype_dim)
        proportion_to_mutate (float): proportion of variables to mutate in
            each genotype (must be in [0, 1]).
        eta: scaling parameter, the larger the more spread the new
            values will be.
        minval: minimum value to clip the genotypes.
        maxval: maximum value to clip the genotypes.

    Returns:
        New genotypes - same shape as input and a new RNG key
    """
    random_key, subkey = jax.random.split(random_key)
    batch_size = jax.tree_leaves(x)[0].shape[0]
    mutation_key = jax.random.split(subkey, num=batch_size)
    mutation_fn = partial(
        _polynomial_mutation,
        proportion_to_mutate=proportion_to_mutate,
        eta=eta,
        minval=minval,
        maxval=maxval,
    )
    mutation_fn = jax.vmap(mutation_fn)
    x = jax.tree_map(lambda x_: mutation_fn(x_, mutation_key), x)
    return x, random_key


def _polynomial_crossover(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    random_key: RNGKey,
    proportion_var_to_change: float,
) -> jnp.ndarray:
    """
    Base crossover for one pair of genotypes.

    x1 and x2 should have the same shape
    In this function we assume x1 shape and x2 shape to be (genotype_dim,)
    """
    num_var_to_change = int(proportion_var_to_change * x1.shape[0])
    indices = jnp.arange(start=0, stop=x1.shape[0])
    selected_indices = jax.random.choice(
        random_key, indices, shape=(num_var_to_change,)
    )
    x = x1.at[selected_indices].set(x2[selected_indices])
    return x


def polynomial_crossover(
    x1: Genotype,
    x2: Genotype,
    random_key: RNGKey,
    proportion_var_to_change: float,
) -> Tuple[Genotype, RNGKey]:
    """
    Crossover over a set of pairs of genotypes.

    Batched version of _simple_crossover_function
    x1 and x2 should have the same shape
    In this function we assume x1 shape and x2 shape to be
    (batch_size, genotype_dim)

    Parameters:
        x1: first batch of genotypes
        x2: second batch of genotypes
        random_key: RNG key for reproducibility
        proportion_var_to_change: proportion of variables to exchange
            between genotypes (must be [0, 1])

    Returns:
        New genotypes and a new RNG key
    """

    random_key, subkey = jax.random.split(random_key)
    batch_size = jax.tree_leaves(x2)[0].shape[0]
    crossover_keys = jax.random.split(subkey, num=batch_size)
    crossover_fn = partial(
        _polynomial_crossover,
        proportion_var_to_change=proportion_var_to_change,
    )
    crossover_fn = jax.vmap(crossover_fn)
    # TODO: check that key usage is correct
    x = jax.tree_map(lambda x1_, x2_: crossover_fn(x1_, x2_, crossover_keys), x1, x2)
    return x, random_key


def isoline_variation(
    x1: Genotype,
    x2: Genotype,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Tuple[Genotype, RNGKey]:
    """
    Iso+Line-DD Variation Operator [1] over a set of pairs of genotypes

    Parameters:
        x1 (Genotypes): first batch of genotypes
        x2 (Genotypes): second batch of genotypes
        random_key (RNGKey): RNG key for reproducibility
        iso_sigma (float): spread parameter (noise)
        line_sigma (float): line parameter (direction of the new genotype)
        minval (float, Optional): minimum value to clip the genotypes
        maxval (float, Optional): maximum value to clip the genotypes

    Returns:
        x (Genotypes): new genotypes
        random_key (RNGKey): new RNG key

    [1] Vassiliades, Vassilis, and Jean-Baptiste Mouret. "Discovering the elite
    hypervolume by leveraging interspecies correlation." Proceedings of the Genetic and
    Evolutionary Computation Conference. 2018.
    """

    # Computing line_noise
    random_key, key_line_noise = jax.random.split(random_key)
    batch_size = jax.tree_leaves(x1)[0].shape[0]
    line_noise = jax.random.normal(key_line_noise, shape=(batch_size,)) * line_sigma

    def _variation_fn(
        x1: jnp.ndarray, x2: jnp.ndarray, random_key: RNGKey
    ) -> jnp.ndarray:
        iso_noise = jax.random.normal(random_key, shape=x1.shape) * iso_sigma
        x = (x1 + iso_noise) + jax.vmap(jnp.multiply)((x2 - x1), line_noise)

        # Back in bounds if necessary (floating point issues)
        if (minval is not None) or (maxval is not None):
            x = jnp.clip(x, minval, maxval)
        return x

    # create a tree with random keys
    nb_leaves = len(jax.tree_leaves(x1))
    random_key, subkey = jax.random.split(random_key)
    subkeys = jax.random.split(subkey, num=nb_leaves)
    keys_tree = jax.tree_unflatten(jax.tree_structure(x1), subkeys)

    # apply isolinedd to each branch of the tree
    x = jax.tree_map(lambda y1, y2, key: _variation_fn(y1, y2, key), x1, x2, keys_tree)

    return x, random_key


class VariationOperator(metaclass=abc.ABCMeta):
    def __init__(self, minval: Optional[float] = None, maxval: Optional[float] = None):
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

        def _get_array_keys(_key: RNGKey, _gen: Genotype) -> jnp.ndarray:
            _key, *_subkeys = jax.random.split(_key, num=_gen.shape[0] + 1)
            return jnp.asarray(_subkeys)

        random_key, _subkey = jax.random.split(random_key)
        print()
        key_arrays_tree = jax.tree_map(
            _get_array_keys,
            VariationOperator.get_tree_keys(genotypes_tree, _subkey)[0],
            genotypes_tree,
        )

        return (
            jax.tree_map(
                jax.vmap(_get_indexes_positions_cross_over),
                genotypes_tree,
                key_arrays_tree,
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


class Normal(Mutation):
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
    v1 = Normal(
        mutation_rate=0.5,
        sigma=0.1,
        minval=None,
        maxval=None,
    )
    v2 = IsolineVariationOperator(
        iso_sigma=0.1,
        line_sigma=0.1,
        cross_over_rate=0.5,
        returns_single_genotype=False,
        minval=None,
        maxval=None,
    )
    v = ComposerVariations(variations_operators_list=[v1, v2], minval=None, maxval=None)
    gen_1 = {"a": jnp.zeros((2, 10))}
    # gen_2 = {"a": jnp.array([[0., -1.], [0.5, 0.5]])}
    random_key = jax.random.PRNGKey(4)
    print(v.apply_with_clip(gen_1, None, random_key)[0])


if __name__ == "__main__":
    _test()
