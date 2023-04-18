# MAP Elites Population Based Training (ME PBT)

[ME PBT](https://openreview.net/forum?id=CBfYffLqWqb) is a recent algorithm combining MAP Elites with Population Based Training to evolve a population of diverse RL agents.

To create an instance of PBTME, one need to use an instance of [Distributed MAP-Elites](map_elites.md) with the PBTEmitter, detailed below.

::: qdax.core.emitters.pbt_me_emitter.PBTEmitter
