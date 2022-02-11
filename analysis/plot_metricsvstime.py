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

METRICS_NAME_TUPLE = (
    DataLoader.ARCHIVE_SIZE,
    DataLoader.BEST_FIT,
    DataLoader.QD_SCORE,
)



def get_interpolate_scores_fn(name_metric): #, fixed_interval_arr):

    def interpolate_scores(data_collection):
        scores_array = data_collection.metrics_df[name_metric].to_numpy()
        epoch_runtime_array = data_collection.timings.epoch_runtime
        scores_array.flatten()
        epoch_runtime_array.flatten()

        fixed_interval_arr = np.arange(0, np.max(epoch_runtime_array), step=20)
        #print("Inside collect_scores: ", epoch_runtime_array)
        return np.interp(
            x=fixed_interval_arr,
            xp=epoch_runtime_array.flatten(),
            fp=scores_array.flatten(),
        )
    
    return interpolate_scores

def get_reduce_truncate_fn(quantile):

    def reduce_truncate(data):
        #print("here 1: ",data)
        data = list(data)
        #print("here 2: ",data)

        # we get arrays of different size for each replication
        min_len = np.inf
        for l in data:
            if len(l) < min_len:
               min_len = len(l)
               #print("min_len: ",min_len)
        new_data = []
        for l in data:
            new_data.append(l[:min_len])    

        print(np.array(new_data).shape)   
        data = np.quantile(new_data, q=quantile, axis=0)

        return data

    return reduce_truncate



def generate_figure(path_save, attribute_comparison_name, data_collector_all_variants):
    sorting_attributes_fn = int

    plt.style.use('classic')
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    pu.set_font_size(13)

    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    axes = np.asarray(axes)
    fig_size = pu.get_fig_size(24, 6)
    fig.set_size_inches(*fig_size)

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
    env_name = list(env_name_per_variant_dict.values())[0]
    grid_shape = list(grid_shape_per_variant_dict.values())[0]
    entire_title_plot = 'Env: ' + str(env_name) + '  Grid shape: ' + str(grid_shape)
    fig.suptitle(entire_title_plot)
    
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

    my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())

    dict_colors_per_variant = {
        variant_folder: my_cmap.colors[index_variant]
        for index_variant, variant_folder in enumerate(attribute_legend_per_variant_ordered)
    }

    # iterate through 3 different metrics - i.e. 3 different plots archive size, best fit, qdscore
    for index_axis in range(0, len(METRICS_NAME_TUPLE)):
        ax = axes.reshape(-1)[index_axis]

        name_metric = METRICS_NAME_TUPLE[index_axis]

        #fixed_interval_arr = np.linspace(start=0, stop=3750, num=100)
        #fixed_interval_arr = np.arange(0, 4000, step=20)
        interpolate_scores_fn = get_interpolate_scores_fn(name_metric=name_metric) #, fixed_interval_arr=fixed_interval_arr)
        reduce_truncate_fn_median = get_reduce_truncate_fn(quantile=0.5)
        reduce_truncate_fn_q1 = get_reduce_truncate_fn(quantile=0.25)
        reduce_truncate_fn_q3 = get_reduce_truncate_fn(quantile=0.75)

        medians_per_variant = data_collector_all_variants.map_and_reduce(
            map_fn=interpolate_scores_fn,
            reduce_fn=reduce_truncate_fn_median
        )
        quartile_1_per_variant = data_collector_all_variants.map_and_reduce(
            map_fn=interpolate_scores_fn,
            reduce_fn=reduce_truncate_fn_q1
        )
        quartile_3_per_variant = data_collector_all_variants.map_and_reduce(
            map_fn=interpolate_scores_fn,
            reduce_fn=reduce_truncate_fn_q3
        )
        
        for index_variant, variant_folder in enumerate(attribute_legend_per_variant_ordered):
            medians = medians_per_variant[variant_folder]
            quantile_1 = quartile_1_per_variant[variant_folder]
            quantile_3 = quartile_3_per_variant[variant_folder]

            #runtime = runtime_per_variant[variant_folder]
            runtime = np.arange(0, len(medians)*20, step=20) #[variant_folder]

            ax.plot(runtime,
                    medians,
                    color=dict_colors_per_variant[variant_folder],
                    zorder=index_variant * 2 + 22,
                    # label=name_legend,
                    # alpha=alpha_median,
                    # linestyle=linestyles[index_variant],
                    # linewidth=linewidth_median,
                    # marker=list_markers[index_variant],
                    # markevery=markevery,
                    )

            ax.fill_between(runtime,
                            quantile_1,
                            quantile_3,
                            alpha=0.3,
                            color=dict_colors_per_variant[variant_folder],
                            zorder=index_variant * 2 + 21,
                            )

        ax.set_title(METRICS_NAME_TUPLE[index_axis])

        ax.grid()

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dict_colors_per_variant[variant], markersize=13)
        for variant in attribute_legend_per_variant_ordered
    ]

    # ax.legend(handles=legend_elements, loc='center')
    fig.legend(legend_elements, list(attribute_legend_per_variant_ordered.values()),
               bbox_to_anchor=(0.5, -0.02),
               ncol=4,
               loc="upper center",
               frameon=False,
               title=attribute_comparison_name)

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
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--attribute', default="population_size",
                        help="which attribute to consider for comparison in config files")
    parser.add_argument('--data-loader', required=True, choices=DataLoader.get_dataloader_from_name_dict().keys())

    return parser.parse_args()


def main():
    args = get_args()
    data_loader = DataLoader.get_dataloader_from_name_dict()[args.data_loader]
    data_collector_all_variants = data_collectors.DataCollectionAllVariants(data_loader=data_loader,
                                                                            main_result_folder=args.results,
                                                                            experiment_name=args.exp_name)
    generate_figure(path_save=args.save,
                    attribute_comparison_name=args.attribute,
                    data_collector_all_variants=data_collector_all_variants,
                    )


if __name__ == "__main__":
    main()
