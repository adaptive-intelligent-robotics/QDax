import jax
import jax.numpy as jnp
import time

"""
Conversion utility - just a first pass which gets the job done. Could probably be optimized or better written.
"""

def list_of_trees_to_single_tree(policy_params_batch, pop_size):
    '''
    policy_params_batch is a list of trees - each tree has a params for a policy model
    pop_size is the number of ind in the list
    
    returns a new tree with tree_def of the policy, but has the params in batch of the pop_size
    '''
    print("Converting list of trees to single tree")
    start = time.time()
    # new_tree = policy_params_batch[0]
    # #jax.tree_map(lambda x: print(x.shape), pparams)
    # new_tree = jax.tree_map(lambda x: jnp.expand_dims(x,axis=0), new_tree)
    # for i in range (1,pop_size):
    #     new_tree = jax.tree_multimap(lambda a, b: jnp.concatenate([a, jnp.expand_dims(b,axis=0)], axis=0), new_tree, policy_params_batch[i])

    new_tree = jax.tree_multimap(lambda *xs: jnp.asarray((xs)), *policy_params_batch)
    print("Time took for conversion: ", time.time() - start)
    return new_tree

def simple_convert_list_to_single(policy_params_batch):
    return jax.tree_multimap(lambda *xs: list(xs), *policy_params_batch)


def single_tree_to_list_of_trees(tree, pop_size):
    '''
    single tree is in a batch form
    
    returns a list of tree with the params
    '''
    print("Converting single tree to list of trees")
    start = time.time()

    new_tree = [jax.tree_map(lambda x: x[i,:], tree) for i in range(pop_size)]

    print("Time took for conversion: ", time.time() - start)

    return new_tree
