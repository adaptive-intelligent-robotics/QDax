import argparse
import operator
from collections import OrderedDict

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


def generate_figure(path_save, attribute_comparison_name, data_collector_all_variants):
    sorting_attributes_fn = int

    plt.style.use("classic")
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    pu.set_font_size(13)

    fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
    fig_size = pu.get_fig_size(9, 6)
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
    env_name = list(env_name_per_variant_dict.values())[0]
    grid_shape = list(grid_shape_per_variant_dict.values())[0]
    entire_title_plot = "Env: " + str(env_name) + "  Grid shape: " + str(grid_shape)
    fig.suptitle(entire_title_plot)

    # GET DATA #
    batch_size_dict = data_collector_all_variants.get_common_attribute_per_variant(
        lambda x: x.configuration.population_size
    )

    final_qd_scores_dict = (
        data_collector_all_variants.map_function_to_replication_collection(
            map_fn=toolz.compose(
                lambda array_np: array_np.flatten()[
                    -1
                ],  # 3. Flattening previously obtained array and getting last value
                lambda df: df[
                    DataLoader.QD_SCORE
                ].to_numpy(),  # 2. Getting QD_Score and converting it to a numpy array
                operator.attrgetter(
                    "metrics_df"
                ),  # 1. Getting metrics_df attribute of DataCollectionOneReplication
            )
        )
    )

    # LEGEND AND COLORS #
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
            key=lambda key_value_pair: sorting_attributes_fn(key_value_pair[1]),
        )
    )
    my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    list_colors = [
        my_cmap.colors[index_color]
        for index_color in range(len(data_collector_all_variants))
    ]

    # MAIN PLOTTTING #
    batch_size_list = []

    final_qd_scores_list = []
    medians_final_qd = []
    quantile_1_final_qd = []
    quantile_3_final_qd = []
    for index_variant, variant_folder in enumerate(
        attribute_legend_per_variant_ordered
    ):

        batch_size_list.append(batch_size_dict[variant_folder])
        final_qd_score = list(final_qd_scores_dict[variant_folder].values())

        final_qd_scores_list.append(final_qd_score)
        medians_final_qd.append(np.quantile(final_qd_score, q=0.5))
        quantile_1_final_qd.append(np.quantile(final_qd_score, q=0.25))
        quantile_3_final_qd.append(np.quantile(final_qd_score, q=0.75))

    print(len(batch_size_list))
    x_ticks = [i for i in range(1, len(batch_size_list) + 1)]

    ax1.plot(x_ticks, medians_final_qd, "o-")
    ax1.fill_between(
        x_ticks,
        quantile_1_final_qd,
        quantile_3_final_qd,
        alpha=0.3,
    )
    ax1.set_xticks(ticks=x_ticks, labels=batch_size_list)
    ax1.set_ylabel("Final QD Score")
    ax1.set_xlabel("Batch Size")
    ax1.grid()

    """
    print(avg_eval_per_sec_list)
    avg_eval_per_sec_arr = np.array(avg_eval_per_sec_list)
    print(avg_eval_per_sec_arr.shape)
    avg_eval_per_sec_arr = np.transpose(avg_eval_per_sec_arr)
    ax1.boxplot(avg_eval_per_sec_arr,)
    ax1.set_xticklabels(batch_size_dict.values())
    """

    # LEGEND FOR PLOT #
    dict_colors_per_variant_name = {
        variant: list_colors[i]
        for i, variant in enumerate(
            data_collector_all_variants.collection_from_variant_str_dict
        )
    }
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dict_colors_per_variant_name[variant],
            markersize=13,
        )
        for variant in data_collector_all_variants.collection_from_variant_str_dict
    ]
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markersize=13),
        *legend_elements,
    ]
    # fig.legend(legend_elements, [attribute_comparison_name] + [attribute_legend_per_variant[variant]
    #                                                            for variant in
    #                                                            data_collector_all_variants.collection_from_variant_str_dict
    #                                                            ],
    #            bbox_to_anchor=(0.5, -0.02),
    #            ncol=4,
    #            loc="upper center",
    #            frameon=False)

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
    parser.add_argument("-s", "--save", help="where to save the results")
    parser.add_argument("--exp-name", type=str)
    parser.add_argument(
        "--attribute",
        default="population_size",
        help="which attribute to consider for comparison in config files",
    )
    parser.add_argument(
        "--data-loader",
        default="qdax",
        choices=DataLoader.get_dataloader_from_name_dict().keys(),
    )

    return parser.parse_args()


def main():
    args = get_args()
    data_loader = DataLoader.get_dataloader_from_name_dict()[args.data_loader]
    data_collector_all_variants = data_collectors.DataCollectionAllVariants(
        data_loader=data_loader,
        main_result_folder=args.results,
        experiment_name=args.exp_name,
    )

    generate_figure(
        path_save=args.save,
        attribute_comparison_name=args.attribute,
        data_collector_all_variants=data_collector_all_variants,
    )


if __name__ == "__main__":
    main()
