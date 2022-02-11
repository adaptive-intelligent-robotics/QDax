import argparse
import glob
import os
from collections import OrderedDict
from operator import attrgetter

import matplotlib
import matplotlib.pyplot as plt
import analysis.data_reader as data_reader

import jax.numpy as jnp



def plot_grid_repertoire(repertoire):
    x_bd = [0,1]
    y_bd = [0,1] 
    fit = [0.2,-1]

    mask = jnp.where(~jnp.isnan(repertoire.fitness),1,0)

    counter = 0
    for r,row in enumerate(mask):
        for c,col in enumerate(row):
            if col>0:
                x_bd.append(r/(repertoire.grid_shape[0]-1))
                y_bd.append(c/(repertoire.grid_shape[1]-1))
                fit.append(repertoire.fitness[(r,c)]/(1000))
                counter+=1

    bd_archive = [x_bd, y_bd]
   

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()
    im = ax.scatter(bd_archive[0], bd_archive[1],c=fit,s=30,cmap=plt.cm.jet,marker='s')
    im.set_clim(0, 1)
    ax.set_aspect(1)
    fig.colorbar(im,ax=ax)

    plt.show()

def main():

    #path_training_state = "ant_omni_10_10_bdtest/qdax_training_env_name-ant_omni_num_epochs-245_episode_length-100_population_size-8192_grid_shape-100-100/2021-12-31_22-47-08_c4ceb9a6-bec4-43a4-9633-57d92fb84635/training_state.pkl"
    #path_training_state = "ant_omni_10_10_bdtest/qdax_training_env_name-ant_omni_num_epochs-245_episode_length-100_population_size-8192_grid_shape-100-100/2021-12-31_22-47-08_c4ceb9a6-bec4-43a4-9633-57d92fb84635/training_state.pkl"
    #path_training_state = "ant_omni_10_10_bdtest/qdax_training_env_name-ant_omni_num_epochs-245_episode_length-100_population_size-8192_grid_shape-100-100/2021-12-31_22-47-08_c4ceb9a6-bec4-43a4-9633-57d92fb84635/training_state.pkl"
    #path_training_state = "ant_omni_10_10_bdtest/qdax_training_env_name-ant_omni_num_epochs-245_episode_length-100_population_size-8192_grid_shape-100-100/2021-12-31_22-47-08_c4ceb9a6-bec4-43a4-9633-57d92fb84635/training_state.pkl"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="where to load the results from")
    args = parser.parse_args()

    training_state = data_reader.load_training_state(args.results_path)
    print("repertoire size: ",training_state.repertoire.num_indivs)

    plot_grid_repertoire(training_state.repertoire)


main()