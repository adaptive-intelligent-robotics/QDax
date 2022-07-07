#!/usr/bin/env python3

import argparse
import os
import subprocess
import tempfile

import build_final_image

EXP_PATH = "git/exp/"
ABSOLUTE_EXP_PATH = "/" + EXP_PATH


def get_default_image_name() -> str:
    return f"{build_final_image.get_project_folder_name()}.sif"


def build_sandbox(path_singularity_def: str, image_name: str) -> None:
    # check if the sandbox has already been created
    if os.path.exists(image_name):
        return

    print(f"{image_name} does not exist, building it now from {path_singularity_def}")
    assert os.path.exists(
        path_singularity_def
    )  # exit if path_singularity_definition_file is not found

    # run commands
    command = (
        f"singularity build --force --fakeroot --sandbox {image_name} "
        f"{path_singularity_def}"
    )
    subprocess.run(command.split())


def run_container(
    nvidia: bool,
    use_no_home: bool,
    use_tmp_home: bool,
    image_name: str,
    binding_folder_inside_container: str,
) -> None:
    additional_args = ""

    if nvidia:
        print("Nvidia runtime ON")
        additional_args += " " + "--nv"

    if use_no_home:
        print("Using --no-home")
        additional_args += " " + "--no-home --containall"

    if use_tmp_home:
        tmp_home_folder = tempfile.mkdtemp(dir="/tmp")
        additional_args += " " + f"--home {tmp_home_folder}"
        build_final_image.error_print(
            f"Warning: The HOME folder is a temporary directory located in "
            f"{tmp_home_folder}! Do not store any result there!"
        )

    if not binding_folder_inside_container:
        binding_folder_inside_container = build_final_image.get_project_folder_name()

    path_folder_binding_in_container = os.path.join(
        image_name, EXP_PATH, binding_folder_inside_container
    )
    if not os.path.exists(path_folder_binding_in_container):
        list_possible_folder_binding_in_container = next(
            os.walk(os.path.join(image_name, EXP_PATH))
        )[1]
        list_possible_options = [
            f"    --binding-folder {existing_folder}"
            for existing_folder in list_possible_folder_binding_in_container
        ]
        build_final_image.error_print(
            f"Warning: The folder "
            f"{os.path.join(ABSOLUTE_EXP_PATH, binding_folder_inside_container)} "
            f"does not exist in the container. The Binding between your project folder "
            f"and your container is likely to be unsuccessful.\n"
            f"You may want to consider adding one of the following options to the "
            f"'start_container' command:\n" + "\n".join(list_possible_options)
        )

    command = (
        f"singularity shell -w {additional_args} "
        f"--bind {os.path.dirname(os.getcwd())}:"
        f"{ABSOLUTE_EXP_PATH}/{binding_folder_inside_container} "
        f"{image_name}"
    )
    subprocess.run(command.split())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sandbox container and shell into it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--nv", action="store_true", help="enable experimental Nvidia support"
    )
    parser.add_argument(
        "--no-home", action="store_true", help='apply --no-home to "singularity shell"'
    )
    parser.add_argument(
        "--tmp-home",
        action="store_true",
        help="binds HOME directory of the singularity container to a temporary folder",
    )

    parser.add_argument(
        "--path-def",
        required=False,
        type=str,
        default=build_final_image.SINGULARITY_DEFINITION_FILE_NAME,
        help="path to singularity definition file",
    )

    parser.add_argument(
        "--personal-token",
        required=False,
        type=str,
        default=build_final_image.get_personal_token(),
        help="Gitlab Personal token. "
        "If not specified, it takes the value of the environment variable "
        "PERSONAL_TOKEN, if it exists. "
        "If the environment variable SINGULARITYENV_PERSONAL_TOKEN is not set yet, "
        "then it is set the value provided.",
    )

    parser.add_argument(
        "-b",
        "--binding-folder",
        required=False,
        type=str,
        default=build_final_image.get_project_folder_name(),
        help=f"If specified, it corresponds to the name folder in {ABSOLUTE_EXP_PATH} "
        f"from which the binding is performed to the current project source code. "
        f"By default, it corresponds to the image name (without the .sif extension)",
    )

    parser.add_argument(
        "-i",
        "--image",
        required=False,
        type=str,
        default=get_default_image_name(),
        help="name of the sandbox image to start",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    enable_nvidia_support = args.nv
    use_no_home = args.no_home
    use_tmp_home = args.tmp_home
    path_singularity_definition_file = args.path_def
    image_name = args.image
    binding_folder_inside_container = args.binding_folder
    personal_token = args.personal_token

    # Create environment variables for singularity
    build_final_image.generate_singularity_environment_variables(
        ci_job_token=None,
        personal_token=personal_token,
        project_folder=binding_folder_inside_container,
    )

    build_sandbox(path_singularity_definition_file, image_name)
    run_container(
        enable_nvidia_support,
        use_no_home,
        use_tmp_home,
        image_name,
        binding_folder_inside_container,
    )


if __name__ == "__main__":
    main()
