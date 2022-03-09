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
    fig_size = pu.get_fig_size(12, 6)
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

    # Gettings medians of full training time per variant:
    medians_full_training_time_dict = data_collector_all_variants.map_and_reduce(
        map_fn=operator.attrgetter("timings.full_training"),
        reduce_fn=lambda x: np.median(np.fromiter(x, dtype=float), axis=0),
    )
    median_final_qd_scores = data_collector_all_variants.map_and_reduce(
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
        ),
        reduce_fn=toolz.compose(
            lambda x: np.median(
                x, axis=0
            ),  # 5. Getting median value of that list of QD scores
            list,  # 4. For each variant, Converting all values of replications into a list (of final QD scores)
        ),
    )
    print("full training time: ", medians_full_training_time_dict)
    print("median_final_qd_score: ", median_final_qd_scores)

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

    dict_colors_per_variant = {
        variant_folder: my_cmap.colors[index_variant]
        for index_variant, variant_folder in enumerate(
            attribute_legend_per_variant_ordered
        )
    }

    # MAIN PLOTTTING #
    for index_variant, variant_folder in enumerate(
        attribute_legend_per_variant_ordered
    ):
        ax1.scatter(
            medians_full_training_time_dict[variant_folder],
            median_final_qd_scores[variant_folder],
            color=dict_colors_per_variant[variant_folder],
            zorder=index_variant * 2 + 22,
        )
        ax1.set_ylabel("Final QD Score")
        ax1.set_xlabel("Runtime")

    # LEGEND FOR PLOT #
    ax1.grid()

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=dict_colors_per_variant[variant],
            markersize=13,
        )
        for variant in attribute_legend_per_variant_ordered
    ]

    fig.legend(
        legend_elements,
        list(attribute_legend_per_variant_ordered.values()),
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        loc="upper center",
        frameon=False,
        title=attribute_comparison_name,
    )

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
