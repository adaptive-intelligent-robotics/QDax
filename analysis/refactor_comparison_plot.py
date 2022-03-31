import operator
from collections import OrderedDict
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import toolz as toolz
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import analysis.plot_utils as pu
from analysis import data_collectors
from analysis.data_loaders import DataLoader


def get_relevant_data(data_collector_all_variants, attribute_comparison_name, sorting_attributes_fn):

    # GET DATA #
    batch_size_dict = data_collector_all_variants.get_common_attribute_per_variant(lambda x: x.configuration.population_size)
    #print(batch_size_dict)

    attribute_legend_per_variant = data_collector_all_variants.map_and_reduce(
        map_fn=toolz.compose(
            operator.attrgetter(attribute_comparison_name),
            operator.attrgetter("configuration"),
        ),
        reduce_fn=toolz.compose(
            operator.itemgetter(0),
            list,
        ),
    )
    # Sorting dictionary legend ordering by sorting attributes using sorting_attributes_fn
    attribute_legend_per_variant_ordered = OrderedDict(
        sorted(
            attribute_legend_per_variant.items(),
            key=lambda key_value_pair: sorting_attributes_fn(key_value_pair[1])
        )
    )

    avg_eval_per_sec_dict = data_collector_all_variants.map_function_to_replication_collection(
        map_fn=operator.attrgetter("timings.avg_eval_per_sec")
    )

    full_runtime_dict = data_collector_all_variants.map_function_to_replication_collection(
        map_fn=operator.attrgetter("timings.full_training")
    )

    batch_size_list = []
    avg_eval_per_sec_list = []
    medians_evalps = []
    quantile_1_evalps = []
    quantile_3_evalps = [] 

    full_runtime_list = []
    medians_full_runtime= []
    quantile_1_full_runtime = []
    quantile_3_full_runtime = [] 
    for index_variant, variant_folder in enumerate(attribute_legend_per_variant_ordered):
        
        batch_size_list.append(batch_size_dict[variant_folder])
        avg_eval_per_sec = list(avg_eval_per_sec_dict[variant_folder].values())
        full_runtime = list(full_runtime_dict[variant_folder].values())

        avg_eval_per_sec_list.append(avg_eval_per_sec)
        medians_evalps.append(np.quantile(avg_eval_per_sec, q=0.5))
        quantile_1_evalps.append(np.quantile(avg_eval_per_sec, q=0.25))
        quantile_3_evalps.append(np.quantile(avg_eval_per_sec, q=0.75))

        full_runtime_list.append(full_runtime)
        medians_full_runtime.append(np.quantile(full_runtime, q=0.5))
        quantile_1_full_runtime.append(np.quantile(full_runtime, q=0.25))
        quantile_3_full_runtime.append(np.quantile(full_runtime, q=0.75))

    return batch_size_list, medians_evalps, quantile_1_evalps, quantile_3_evalps, medians_full_runtime, quantile_1_full_runtime, quantile_3_full_runtime


def generate_figure(path_save, attribute_comparison_name, 
                    data_collector_all_variants, 
                    data_collector_all_variants_2,
                    data_collector_all_variants_3,
                    data_collector_all_variants_4,
                    data_collector_all_variants_5,
                    data_collector_all_variants_6,
                    ):
    sorting_attributes_fn = int

    plt.style.use('classic')
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    pu.set_font_size(13)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    fig_size = pu.get_fig_size(14, 6)
    fig.set_size_inches(*fig_size)
    
    # plot env_name and grid shape in title
    env_name_per_variant_dict = data_collector_all_variants.map_and_reduce(
        map_fn=toolz.compose(
            operator.attrgetter("configuration.env_name"),
        ),
        reduce_fn=toolz.compose(
            operator.itemgetter(0),
            list,
        ),
        )
    grid_shape_per_variant_dict = data_collector_all_variants.map_and_reduce(
        map_fn=toolz.compose(
            operator.attrgetter("configuration.grid_shape"),
        ),
        reduce_fn=toolz.compose(
            operator.itemgetter(0),
            list,
        ),
        )
    #print(env_name_per_variant_dict)
    env_name = list(env_name_per_variant_dict.values())[0]
    grid_shape = list(grid_shape_per_variant_dict.values())[0]
    #entire_title_plot = 'Env: ' + str(env_name) + '  Grid shape: ' + str(grid_shape)
    #fig.suptitle(entire_title_plot)

    # LEGEND AND COLORS #
    
    my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())

    implementation_variants = ["QDax-3060 (Refactor)", 
                               "QDax-2080 (Refactor)", 
                               "QDax-3060 (Ori)", 
                               "QDax-2080 (Ori)",
                               "QDax-3060 (Refactor Vec)", 
                               "QDax-2080 (Refactor Vec)"]
    list_colors = [
        my_cmap.colors[index_color] for index_color in range(len(implementation_variants))
    ]
    dict_colors_per_variant_name = {
        variant: list_colors[i]
        for i, variant in enumerate(implementation_variants)
    }

    # MAIN PLOTTTING #
    ## RIGHT PLOT
    data_collector_list = [data_collector_all_variants, 
                           data_collector_all_variants_2,
                           data_collector_all_variants_3,
                           data_collector_all_variants_4,
                           data_collector_all_variants_5,
                           data_collector_all_variants_6,
                           ]
    for idx, data_collector_av in enumerate(data_collector_list):

        batch_size_list, medians_evalps, quantile_1_evalps, quantile_3_evalps, \
        medians_full_runtime, quantile_1_full_runtime, quantile_3_full_runtime = get_relevant_data(data_collector_av, attribute_comparison_name, sorting_attributes_fn)
    
        #print(len(batch_size_list))
        print("Max median evalps of variant: ",np.max(medians_evalps))
        print("ArgMax median evalps of variant: ",np.argmax(medians_evalps))
        x_ticks = [i for i in range(1, len(batch_size_list)+1)]
        
        ax1.plot(batch_size_list, medians_evalps, 'o-', color=dict_colors_per_variant_name[implementation_variants[idx]])
        ax1.fill_between(batch_size_list,
                                quantile_1_evalps,
                                quantile_3_evalps,
                                alpha=0.3,
                                color=dict_colors_per_variant_name[implementation_variants[idx]])
    
    #ax1.set_xticks(ticks=x_ticks, labels=batch_size_list) 
    ax1.set_ylabel("Average Eval/s")
    ax1.set_xlabel("Batch Size")
    #ax1.set_yscale('log') 
    ax1.set_xscale('log') 
    ax1.grid()

    # ## LEFT PLOT
    data_collector_list_2 = [data_collector_all_variants, 
                           data_collector_all_variants_2,
                           data_collector_all_variants_3,
                           data_collector_all_variants_4,
                           data_collector_all_variants_5,
                           data_collector_all_variants_6,]
    for idx, data_collector_av in enumerate(data_collector_list_2):

        batch_size_list, medians_evalps, quantile_1_evalps, quantile_3_evalps, \
        medians_full_runtime, quantile_1_full_runtime, quantile_3_full_runtime = get_relevant_data(data_collector_av, attribute_comparison_name, sorting_attributes_fn)
    
        #print(len(batch_size_list))
        print("Min median full runtime of variant: ",np.min(medians_full_runtime))
        print("ArgMin median full runtime of variant: ",np.argmin(medians_full_runtime))
        x_ticks = [i for i in range(1, len(batch_size_list)+1)]

        ax2.plot(batch_size_list, medians_full_runtime, 'o-', color=dict_colors_per_variant_name[implementation_variants[idx]])
        ax2.fill_between(batch_size_list,
                                quantile_1_full_runtime,
                                quantile_3_full_runtime,
                                alpha=0.3,
                                color=dict_colors_per_variant_name[implementation_variants[idx]])

    #ax2.set_xticks(ticks=x_ticks, labels=batch_size_list) 
    ax2.set_ylabel("Full Runtime (s)")
    ax2.set_xlabel("Batch Size")
    ax2.set_yscale('log') 
    ax2.set_xscale('log')
    ax2.grid()  

    # LEGEND FOR PLOT #
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dict_colors_per_variant_name[variant], markersize=13)
        for variant in implementation_variants
    ]
    # CAREFUL THIS IS ONLY IF YOU HAVE THE TITLE FOR THE LEGENDS - SO WE PUT A WHITE MARKER SO IT IS NOT VISIBLE
    # REMOVE THIS PART OF CODE IF NOT USING TITLE FOR LEGENDS
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor="w", markersize=13),
    #     *legend_elements
    # ]
    fig.legend(legend_elements, [variant
                                for variant in
                                implementation_variants
                                ],
               bbox_to_anchor=(0.5, -0.02),
               ncol=3,
               loc="upper center",
               frameon=False)


    # SAVING OR SHOWING PLOT #
    if path_save:
        pu.save_fig(fig, path_save)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, help="where to load the results from")
    parser.add_argument('-s', '--save', help="where to save the results")
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--attribute', default="population_size",
                        help="which attribute to consider for comparison in config files")
    parser.add_argument('--data-loader', default='qdax', choices=DataLoader.get_dataloader_from_name_dict().keys())

    return parser.parse_args()

def main():
    args = get_args()
    qdax_data_loader = DataLoader.get_dataloader_from_name_dict()['qdax'] #[args.data_loader]

    ## LEFT SUBPLOT DATA - FIXED EPOCHS ##
    gpu_3060_qdax_data_collector_all_variants_refactor = data_collectors.DataCollectionAllVariants(data_loader=qdax_data_loader,
                                                                            main_result_folder="./refactor_results_3060/", #args.results,
                                                                            experiment_name='qdax_training',
                                                                            #load_results_directly=False,
                                                                            )#args.exp_name)
                                                                        
    gpu_2080_qdax_data_collector_all_variants_refactor = data_collectors.DataCollectionAllVariants(data_loader=qdax_data_loader,
                                                                            main_result_folder="./refactor_results_2080/",#"./results/2021-12-14_21_09_06_660105_fAL/", #args.results,
                                                                            experiment_name='qdax_training',
                                                                            #load_results_directly=False,
                                                                            )#args.exp_name)

    gpu_3060_qdax_data_collector_all_variants_original = data_collectors.DataCollectionAllVariants(data_loader=qdax_data_loader,
                                                                            main_result_folder="./replicate_paper_results_3060/", #args.results,
                                                                            experiment_name='qdax_training',
                                                                            #load_results_directly=False,
                                                                            )#args.exp_name)
                                                                        
    gpu_2080_qdax_data_collector_all_variants_original = data_collectors.DataCollectionAllVariants(data_loader=qdax_data_loader,
                                                                            main_result_folder="./replicate_paper_results_2080/",#"./results/2021-12-14_21_09_06_660105_fAL/", #args.results,
                                                                            experiment_name='qdax_training',
                                                                            #load_results_directly=False,
                                                                            )#args.exp_name)

    gpu_3060_qdax_data_collector_all_variants_refactor_vec = data_collectors.DataCollectionAllVariants(data_loader=qdax_data_loader,
                                                                            main_result_folder="./refactor_results_vec_3060/", #args.results,
                                                                            experiment_name='qdax_training',
                                                                            #load_results_directly=False,
                                                                            )#args.exp_name)
                                                                        
    gpu_2080_qdax_data_collector_all_variants_refactor_vec = data_collectors.DataCollectionAllVariants(data_loader=qdax_data_loader,
                                                                            main_result_folder="./refactor_results_vec_2080/",#"./results/2021-12-14_21_09_06_660105_fAL/", #args.results,
                                                                            experiment_name='qdax_training',
                                                                            #load_results_directly=False,
                                                                            )#args.exp_name)
    
    # gpu_2080_qdax_data_collector_all_variants_refactor.load_results_parallel()
    # gpu_3060_qdax_data_collector_all_variants_refactor.load_results_parallel()
    # gpu_2080_qdax_data_collector_all_variants_original.load_results_parallel()
    # gpu_3060_qdax_data_collector_all_variants_original.load_results_parallel()
    

    generate_figure(path_save=args.save,
                    attribute_comparison_name=args.attribute,
                    data_collector_all_variants=gpu_3060_qdax_data_collector_all_variants_refactor, 
                    data_collector_all_variants_2=gpu_2080_qdax_data_collector_all_variants_refactor,
                    data_collector_all_variants_3=gpu_3060_qdax_data_collector_all_variants_original, 
                    data_collector_all_variants_4=gpu_2080_qdax_data_collector_all_variants_original,
                    data_collector_all_variants_5=gpu_3060_qdax_data_collector_all_variants_refactor_vec, 
                    data_collector_all_variants_6=gpu_2080_qdax_data_collector_all_variants_refactor_vec,
                    )
                    #data_collector_all_variants_3=pyme_data_collector_all_variants,)

if __name__ == "__main__":
    main()

