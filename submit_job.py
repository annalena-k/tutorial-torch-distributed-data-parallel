import argparse
import os
import sys
import yaml


def create_submission_file(
        out_dir: str,
        condor_settings: dict,
        filename: str = "submission_file.sub"
):
    """
    Automatically creates submission file.

    Parameters
    ----------
    out_dir: str
        Path to directory in which the submission file is written and where the condor files (.err, .log, .out) are logged.
    condor_settings: dict
        Contains condor settings.
    filename: str
        Name of submission file.
    """
    lines = []
    lines.append(f'executable = {condor_settings["executable"]}\n')
    lines.append(f'request_cpus = {condor_settings["num_cpus"]}\n')
    lines.append(f'request_memory = {condor_settings["memory_cpus"]}\n')
    if "num_gpus" in condor_settings:
        lines.append(f'request_gpus = {condor_settings["num_gpus"]}\n')
    if "memory_gpus" in condor_settings:
        lines.append(
            f"requirements = TARGET.CUDAGlobalMemoryMb > "
            f'{condor_settings["memory_gpus"]}\n\n'
        )
    lines.append(f'arguments = \"{condor_settings["arguments"]}\"\n')
    lines.append(f'error = {os.path.join(out_dir, "info.err")}\n')
    lines.append(f'output = {os.path.join(out_dir, "info.out")}\n')
    lines.append(f'log = {os.path.join(out_dir, "info.log")}\n')
    lines.append("queue")

    with open(os.path.join(out_dir, filename), "w") as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit job based on settings.yaml file.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Path to settings.yaml file.",
    )
    args = parser.parse_args()

    # Read in settings file
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    # Create folder to which the submission file is written
    out_dir = settings["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create submission file
    submission_file = "submission_file.sub"
    condor_settings = settings["local"]["condor"]
    condor_settings["arguments"] = f"{settings['script_path']} --settings_file {args.settings_file}"
    condor_settings["executable"] = sys.executable
    create_submission_file(out_dir=out_dir, condor_settings=condor_settings, filename=submission_file)

    bid = condor_settings["bid"]
    os.system(f"condor_submit_bid {bid} " f"{os.path.join(out_dir, submission_file)}")
