"""Implementation of an updated version of the algorithm QDPG presented in the
paper https://arxiv.org/abs/2006.08505.

QDPG has been udpated to enter in the container+emitter framework of QD. Furthermore,
it has been updated to work better with Jax in term of time cost. Those changes have
been made in accordance with the authors of this algorithm.
"""
import functools
from dataclasses import dataclass
from typing import Callable

import flax.linen as nn

from qdax.core.containers.archive import Archive
from qdax.core.emitters.dpg_emitter import DiversityPGConfig, DiversityPGEmitter
from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Reward, StateDescriptor


@dataclass
class QDPGEmitterConfig:
    qpg_config: QualityPGConfig
    dpg_config: DiversityPGConfig
    iso_sigma: float
    line_sigma: float
    ga_batch_size: int


class QDPGEmitter(MultiEmitter):
    def __init__(
        self,
        config: QDPGEmitterConfig,
        policy_network: nn.Module,
        env: QDEnv,
        score_novelty: Callable[[Archive, StateDescriptor], Reward],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env

        # define the quality emitter
        q_emitter = QualityPGEmitter(
            config=config.qpg_config, policy_network=policy_network, env=env
        )
        # define the diversity emitter
        d_emitter = DiversityPGEmitter(
            config=config.dpg_config,
            policy_network=policy_network,
            env=env,
            score_novelty=score_novelty,
        )

        # define the GA emitter
        variation_fn = functools.partial(
            isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
        )
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=config.ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, d_emitter, ga_emitter))
