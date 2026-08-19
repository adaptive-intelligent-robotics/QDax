# MONET

[MONET](https://arxiv.org/abs/2604.21991) (Multi-Task Optimization over
Networks of Tasks) is a graph-based multi-task optimization algorithm: the
tasks are the nodes of a similarity graph and each node stores the best
solution found so far for its task. At every step, the elite of a random focal
task is either mutated (individual learning) or crossed-over with the elite of
one of the neighbors of the task in the graph (social learning), and the
offspring is evaluated on the focal task.

::: qdax.core.monet.MONET
