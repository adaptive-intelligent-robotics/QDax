import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison


def generate_p_values_log_files(
    df,
    gen,
    environment,
    list_metrics,
    path_file="p-values.log",
    comparison_on_metric="name_variant",
):
    df = df[df["environment"] == environment]
    if gen:
        df = df[df["gen"] == gen]

    df = df[~df[comparison_on_metric].isnull()]

    if not df.empty:

        with open(path_file, "w") as f:

            for metric in list_metrics:
                print(df[metric], list(df[comparison_on_metric]))
                MultiComp = MultiComparison(df[metric], df[comparison_on_metric])
                MultiComp.decimal_tvalues = 10
                comp = MultiComp.allpairtest(
                    stats.ranksums,
                    method="Holm",
                )
                print(f"------ {metric} -------")
                print(comp[0])
                print(f"=======================")

                f.write(f"------ {metric} -------\n")
                f.write(f"{comp[0]}\n")

                for (data, adjusted_p_value) in zip(comp[2], comp[1][2]):
                    str_info = f"{data[0]} - {data[1]} ---> {adjusted_p_value}"
                    print(str_info)
                    f.write(f"{str_info}\n")

                f.write(f"=======================\n\n")


def main():
    # df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")
    df_hexapod_suppl_obs = pd.read_csv(
        "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_02_analysis_learned_behavioural_space/data_obs_hexapod/df_hexapod_camera_vertical_processed.csv"
    )

    # for exp in [
    #     maze_experiments.MAZE_AURORA_SURPRISE_10_COLORS,
    #     air_hockey_experiments.AIR_HOCKEY_AURORA_10_COLORS_SURPRISE,
    #     hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE,
    # ]:
    #     df.loc[df[analysis.metrics.air_hockey.MetricsAirHockey.MAIN_FOLDER] == exp.get_results_folder_name(), "name_variant"] = "aurora_surprise_10_psat"
    #     # exp_without_fit = copy.deepcopy(exp)
    #     # exp_without_fit._has_fit = False
    #     # print("bouh", exp_without_fit.get_results_folder_name())
    #     # df.loc[df[analysis.metrics.air_hockey.MetricsAirHockey.MAIN_FOLDER] == exp_without_fit.get_results_folder_name(), "name_variant"] = "aurora_surprise_10_psat"
    #

    # list_algorithms = [
    #     "aurora_uniform_10_psat",
    #     "aurora_novelty_10_psat",
    #     "aurora_surprise_10_psat", # TODO
    #     "TAXONS_10",
    #     "qd_uniform_psat",
    #     "qd_no_selection_psat",
    #     "NS",
    # ]
    #
    # df = df[df["name_variant"].isin(list_algorithms)]
    #
    # print(df)
    #
    # generate_p_values_log_files(df, gen=15000, environment="hexapod_camera_vertical",
    #                             list_metrics=["mean_fitness", "coverage_pos_40", "angle_coverage"],
    #                             path_file="p_values/p-values-hexapod.log")
    #
    # generate_p_values_log_files(df, gen=1000, environment="air_hockey",
    #                             list_metrics=["mean_fitness", "coverage_pos_40", "diversity"],
    #                             path_file="p_values/p-values-air-hockey.log")
    #
    # generate_p_values_log_files(df, gen=10000, environment="maze",
    #                             list_metrics=["mean_fitness", "coverage_pos_40"],
    #                             path_file="p_values/p-values-maze.log")
    #
    # generate_p_values_log_files(df, gen=1000, environment="maze",
    #                             list_metrics=["mean_fitness", "coverage_pos_40"],
    #                             path_file="p_values/p-values-maze-gen-1000.log")
    #
    #
    # generate_p_values_log_files(df, gen=1000, environment="hexapod_camera_vertical",
    #                             list_metrics=["mean_fitness", "coverage_pos_40"],
    #                             path_file="p_values/p-values-hexapod_camera_vertical-gen-1000.log")
    #
    # print(df[df["environment"] == "maze"][df["name_variant"] == "qd_uniform_psat"][df["gen"] == 10000]["coverage_pos_40"].min())
    # print(df[df["environment"] == "maze"][df["name_variant"] == "aurora_uniform_10_psat"][df["gen"] == 10000]["coverage_pos_40"].max())

    list_algorithms_hexa_diversity = [
        "qd_uniform_psat",
        "aurora_uniform_2_psat",
        "aurora_uniform_3_psat",
        # "aurora_uniform_4_psat",
        "aurora_uniform_5_psat",
        # "aurora_uniform_8_psat",
        "aurora_uniform_10_psat",
    ]
    df_hexapod_suppl_obs = df_hexapod_suppl_obs[
        df_hexapod_suppl_obs["name_variant"].isin(list_algorithms_hexa_diversity)
    ]

    generate_p_values_log_files(
        df_hexapod_suppl_obs,
        gen=15000,
        environment="hexapod_camera_vertical",
        list_metrics=["angle_coverage"],
        path_file="p-values-diversity-hexa.log",
    )


if __name__ == "__main__":
    main()
