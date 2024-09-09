# MAP Elites class

This class implement the base mechanism of MAP-Elites. It must be used with an emitter. To get the usual MAP-Elites algorithm, one must use the [mixing emitter](emitters.md#qdax.core.emitters.standard_emitters.MixingEmitter).

The MAP-Elites class can be used with other emitters to create variants, like [PGAME](pgame.md), [DCRL-ME](dcrlme.md) [CMA-MEGA](cma_mega.md) and [OMG-MEGA](omg_mega.md).

::: qdax.core.map_elites.MAPElites

We also provide a class to have MAP-Elites efficiently distributed over several devices.

::: qdax.core.distributed_map_elites.DistributedMAPElites
