# Covariance Matrix Adaptation MAP Elites (CMAME)

To create an instance of CMAME, one need to use an instance of [MAP-Elites](map_elites.md) with the desired CMA Emitter - optimizing, random direction, improvement - detailed below.To use the pool of emitter mechanism, use the CMAPoolEmitter.

Three emitter types:

::: qdax.core.emitters.cma_emitter.CMAEmitter
::: qdax.core.emitters.cma_rnd_emitter.CMARndEmitter
::: qdax.core.emitters.cma_opt_emitter.CMAOptimizingEmitter

Pool of homogeneous emitters:

::: qdax.core.emitters.cma_pool_emitter.CMAPoolEmitter
