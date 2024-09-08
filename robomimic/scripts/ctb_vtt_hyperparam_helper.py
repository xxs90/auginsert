"""
A useful script for generating json files and shell scripts for conducting parameter scans.
The script takes a path to a base json file as an argument and a shell file name.
It generates a set of new json files in the same folder as the base json file, and 
a shell file script that contains commands to run for each experiment.

Instructions:

(1) Start with a base json that specifies a complete set of parameters for a single 
    run. This only needs to include parameters you want to sweep over, and parameters
    that are different from the defaults. You can set this file path by either
    passing it as an argument (e.g. --config /path/to/base.json) or by directly
    setting the config file in @make_generator. The new experiment jsons will be put
    into the same directory as the base json.

(2) Decide on what json parameters you would like to sweep over, and fill those in as 
    keys in @make_generator below, taking note of the hierarchical key
    formatting using "/" or ".". Fill in corresponding values for each - these will
    be used in creating the experiment names, and for determining the range
    of values to sweep. Parameters that should be sweeped together should
    be assigned the same group number.

(3) Set the output script name by either passing it as an argument (e.g. --script /path/to/script.sh)
    or by directly setting the script file in @make_generator. The script to run all experiments
    will be created at the specified path.

Args:
    config (str): path to a base config json file that will be modified to generate config jsons.
        The jsons will be generated in the same folder as this file.

    script (str): path to output script that contains commands to run the generated training runs

Example usage:

    # assumes that /tmp/gen_configs/base.json has already been created (see quickstart section of docs for an example)
    python hyperparam_helper.py --config /tmp/gen_configs/base.json --script /tmp/gen_configs/out.sh
"""
import argparse

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file, wandb_proj_name="ctb_modality_ablation_concat", num_splits=args.num_splits
    )

    generator.add_param(
        key="train.data",
        name="dataset",
        group=0,
        values=[
            # "ctb_data/datasets/canonical.hdf5",
            # "ctb_data/datasets/variations.hdf5",
            # "ctb_data/datasets/clone_stacked2.hdf5",
            "ctb_data/datasets/clone_stacked6.hdf5",
            # "ctb_data/datasets/clone_stacked12.hdf5"
        ],
        value_names=[
            # "canonical",
            # "nonclone",
            # "vtail2",
            "vtail6",
            # "vtail12"
        ]
    )

    generator.add_param(
        key="algo.ail.enabled",
        name="",
        group=0,
        values=[True],
    )

    generator.add_param(
        key="observation.modalities.obs.low_dim",
        name="prop",
        group=1,
        values=[
            [],
            ["robot0_robot1_proprioception-state"]
        ],
        value_names=["off", "on"]
    )

    generator.add_param(
        key="observation.modalities.obs.ft",
        name="ft",
        group=2,
        values=[
            [],
            ["robot0_robot1_forcetorque-state"]
        ],
        value_names=["off", "on"]
    )

    # generator.add_param(
    #     key="algo.ail.policy_pretrain_epochs",
    #     name="type",
    #     group=1,
    #     values=[40,0],
    #     value_names=["canon-frozen", "noncanon-frozen"]
    # )

    # generator.add_param(
    #     key="algo.ail.latent_pretrain_epochs",
    #     name="",
    #     group=1,
    #     values=[60,0]
    # )

    # generator.add_param(
    #     key="algo.ail.latent_train_epochs",
    #     name="type",
    #     group=1,
    #     values=[50],
    #     value_names=["noncanon"]
    # )

    # generator.add_param(
    #     key="experiment.rollout.warmstart",
    #     name="",
    #     group=1,
    #     values=[49]
    # )

    # generator.add_param(
    #     key="train.num_epochs",
    #     name="",
    #     group=1,
    #     values=[150]
    # )

    # generator.add_param(
    #     key="algo.vtt.vtt_kwargs.token_drop_rate",
    #     name="",
    #     group=2,
    #     values=[0.0,0.5,0.5],
    #     # value_names=["on"]
    # )

    # generator.add_param(
    #     key="algo.vtt.vtt_kwargs.modality_dropout",
    #     name="tokendrop",
    #     group=2,
    #     values=[False,True,False],
    #     value_names=["none","modality","random"]
    # )

    generator.add_param(
        key="train.seed",
        name="seed",
        group=3,
        values=[
            2023,2024,2025,2026,2027,2028
        ],
        value_names=["2023","2024","2025","2026","2027","2028"]
    )

    return generator


def main(args):

    # make config generator
    generator = make_generator(config_file=args.config, script_file=args.script)

    # generate jsons and script
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )

    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="number of separate run files to split up json paths to"
    )

    args = parser.parse_args()
    main(args)
