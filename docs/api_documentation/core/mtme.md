# Multi-task MAP-Elites (MT-ME)

[MT-ME](https://arxiv.org/abs/2003.04407) is a variant of MAP-Elites for
multi-task optimization, i.e. when the fitness function depends on the task and
must be evaluated separately for each task. Each cell of the repertoire
corresponds to a task and stores the best solution found for this task. The
task on which an offspring is evaluated is selected with a tournament biased
towards the task of its first parent, whose size is controlled by a UCB1
multi-armed bandit.

::: qdax.core.mtme.MultiTaskMAPElites
