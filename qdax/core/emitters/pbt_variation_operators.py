from typing import Tuple

from qdax.baselines.sac_pbt import PBTSacTrainingState
from qdax.baselines.td3_pbt import PBTTD3TrainingState
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.types import RNGKey


def sac_pbt_variation_fn(
    training_state1: PBTSacTrainingState,
    training_state2: PBTSacTrainingState,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
) -> Tuple[PBTSacTrainingState, RNGKey]:
    """
    This operator runs a cross-over between two SAC agents. It is used as variation
    operator in the SAC-PBT-Map-Elites algorithm. An isoline-dd variation is applied
    to policy networks, critic networks and entropy alpha coefficients.

    Args:
        training_state1: Training state of first SAC agent.
        training_state2: Training state of first SAC agent.
        random_key: Random key.
        iso_sigma: Spread parameter (noise).
        line_sigma: Line parameter (direction of the new genotype).

    Returns:
        A new SAC training state obtained from cross-over and an updated random key.

    """

    policy_params1, policy_params2 = (
        training_state1.policy_params,
        training_state2.policy_params,
    )
    critic_params1, critic_params2 = (
        training_state1.critic_params,
        training_state2.critic_params,
    )
    alpha_params1, alpha_params2 = (
        training_state1.alpha_params,
        training_state2.alpha_params,
    )
    (policy_params, critic_params, alpha_params), random_key = isoline_variation(
        x1=(policy_params1, critic_params1, alpha_params1),
        x2=(policy_params2, critic_params2, alpha_params2),
        random_key=random_key,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
    )

    new_training_state = training_state1.replace(
        policy_params=policy_params,
        critic_params=critic_params,
        alpha_params=alpha_params,
    )

    return (
        new_training_state,
        random_key,
    )


def td3_pbt_variation_fn(
    training_state1: PBTTD3TrainingState,
    training_state2: PBTTD3TrainingState,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
) -> Tuple[PBTTD3TrainingState, RNGKey]:
    """
    This operator runs a cross-over between two TD3 agents. It is used as variation
    operator in the TD3-PBT-Map-Elites algorithm. An isoline-dd variation is applied
    to policy networks and critic networks.

    Args:
        training_state1: Training state of first TD3 agent.
        training_state2: Training state of first TD3 agent.
        random_key: Random key.
        iso_sigma: Spread parameter (noise).
        line_sigma: Line parameter (direction of the new genotype).

    Returns:
        A new TD3 training state obtained from cross-over and an updated random key.

    """

    policy_params1, policy_params2 = (
        training_state1.policy_params,
        training_state2.policy_params,
    )
    critic_params1, critic_params2 = (
        training_state1.critic_params,
        training_state2.critic_params,
    )
    (policy_params, critic_params,), random_key = isoline_variation(
        x1=(policy_params1, critic_params1),
        x2=(policy_params2, critic_params2),
        random_key=random_key,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
    )
    new_training_state = training_state1.replace(
        policy_params=policy_params,
        critic_params=critic_params,
    )

    return (
        new_training_state,
        random_key,
    )
