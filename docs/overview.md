# QDax Overview

QDax has been designed to be modular yet flexible so it's easy for anyone to use and extend on the different state-of-the-art QD algortihms available.
For instance, MAP-Elites is designed to work with a few modular and simple components: `container`, `emitter`, and `scoring_function`.

## Key concepts
### Container
The `container` specifies the structure of archive of solutions to keep and the addition conditions associated with the archive.

### Emitter
The `emitter` component is responsible for generating new solutions to be evaluated. For example, new solutions can be generated with random mutations, gradient descent, or sampling from distributions as in evolutionary strategies.

### Scoring Function
The `scoring_function` defines the problem/task we want to solve and functions to evaluate the solutions. For example, the `scoring_function` can be used to represent standard black-box optimization tasks such as rastrigin or RL tasks.

## Design Choices
With this modularity, a user can easily swap out any one of the components and pass it to the `MAPElites` class, avoiding having to re-implement all the steps of the algorithm.

Under one layer of abstraction, users have a bit more flexibility. QDax has similarities to the simple and commonly found `ask`/`tell` interface. The `ask` function is similar to the `emit` function in QDax and the `tell` function is similar to the `update` function in QDax. Likewise, the `eval` of solutions is analogous to the `scoring function` in QDax.
More importantly, QDax handles the archive management which is the key idea of QD algorihtms and not present or needed in standard optimization algorihtms or evolutionary strategies.

## Code Example
```python
# Initializes repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init(init_variables, centroids, random_key)

for i in range(num_iterations):

    # generate new population with the emitter
    genotypes, random_key = map_elites._emitter.emit(
        repertoire, emitter_state, random_key
    )

    # scores/evaluates the population
    fitnesses, descriptors, extra_scores, random_key = map_elites._scoring_function(
        genotypes, random_key
    )

    # update repertoire
    repertoire = repertoire.add(genotypes, descriptors, fitnesses)

    # update emitter state
    emitter_state = map_elites._emitter.state_update(
        emitter_state=emitter_state,
        repertoire=repertoire,
        genotypes=genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        extra_scores=extra_scores,
    )
```
