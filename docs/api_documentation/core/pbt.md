# Population Based Training (PBT)

[PBT](https://arxiv.org/abs/1711.09846) is optimization method to jointly optimise a population of models and their hyperparameters to maximize performance.

To use PBT in QDax to train SAC, one can use the two following components (see [examples](../../examples/sac_pbt.ipynb) to see how to use the components appropriatly):

::: qdax.baselines.sac_pbt.PBTSAC

and

::: qdax.baselines.pbt.PBT

To use PBT in order to train TD3 agents, please use the PBTTD3 class:

::: qdax.baselines.td3_pbt.PBTTD3
