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
        base_config_file=config_file, script_file=script_file, wandb_proj_name=args.wandb_proj_name, num_splits=args.num_splits
    )

    # modality input experiment
    # generator.add_param(
    #     key="train.data",
    #     name="exp",
    #     group=0,
    #     values=[
    #         "ctb_data/datasets/train_wrist_canonical.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5"
    #     ],
    #     value_names=[
    #         "canonical",
    #         "tclone6"
    #     ]
    # )

    # generator.add_param(
    #     key="train.output_dir",
    #     name="",
    #     group=0,
    #     values=[
    #         "../experiments/ablation_wristviews_canonical",
    #         "../experiments/ablation_wristviews_tclone"
    #     ]
    # )

    # generator.add_param(
    #     key="train.num_traj_clones",
    #     name="",
    #     group=0,
    #     values=[
    #         0,
    #         6
    #     ]
    # )

    generator.add_param(
        key="observation.modalities.obs.low_dim",
        name="prop",
        group=0,
        values=[
            ["robot0_robot1_proprioception-state"], # no vision
            ["robot0_robot1_proprioception-state"], # no touch
            [], # no proprioception
            [], # vision only
            ["robot0_robot1_proprioception-state"] # all
        ],
        value_names=[
            "on",
            "on",
            "off",
            "off",
            "on"
        ]
    )

    generator.add_param(
        key="observation.modalities.obs.ft",
        name="ft",
        group=0,
        values=[
            ["robot0_robot1_forcetorque-state"],
            [],
            ["robot0_robot1_forcetorque-state"],
            [],
            ["robot0_robot1_forcetorque-state"]
        ],
        value_names=[
            "on",
            "off",
            "on",
            "off",
            "on"
        ]
    )

    generator.add_param(
        key="observation.modalities.obs.rgb",
        name="rgb-wrist",
        group=0,
        values=[
            [],
            ["left_wristview_image", "right_wristview_image"],
            ["left_wristview_image", "right_wristview_image"],
            ["left_wristview_image", "right_wristview_image"],
            ["left_wristview_image", "right_wristview_image"],
        ],
        value_names=[
            "off",
            "on",
            "on",
            "on",
            "on"
        ]
    )

    # generator.add_param(
    #     key="observation.modalities.obs.depth",
    #     name="",
    #     group=1,
    #     values=[
    #         [],
    #         ["overhead_depth"],
    #         ["overhead_depth"],
    #         ["overhead_depth"],
    #         ["overhead_depth"]
    #     ]
    # )

    # # num clones experiment
    # generator.add_param(
    #     key="train.data",
    #     name="",
    #     group=0,
    #     values=[
    #         "ctb_data/datasets/train_wrist_canonical.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5",
    #         "ctb_data/datasets/train_wrist_clone12.hdf5"
    #     ]
    # )

    # generator.add_param(
    #     key="train.num_traj_clones",
    #     name="exp",
    #     group=0,
    #     values=[
    #         0,2,4,6,8,10,12
    #     ],
    #     value_names=[
    #         "0clones",
    #         "2clones",
    #         "4clones",
    #         "6clones",
    #         "8clones",
    #         "10clones",
    #         "12clones",
    #     ]
    # )

    # token dropout experiments
    # generator.add_param(
    #     key="algo.vtt.vtt_kwargs.token_drop_rate",
    #     name="exp",
    #     group=0,
    #     values=[0.0,0.2,0.4,0.6,0.8],
    #     value_names=["drop00","drop20","drop40","drop60","drop80"]
    # )

    # # training set experiments
    # generator.add_param(
    #     key="train.data",
    #     name="dataset",
    #     group=0,
    #     values=[
    #         # "ctb_data/datasets/train_wrist_canonical.hdf5",
    #         # "ctb_data/datasets/train_wrist_clone12.hdf5",
    #         # "ctb_data/datasets/train_swap_clone12.hdf5",
    #         # "ctb_data/datasets/train_vis_wrist_clone12.hdf5",
    #         # "ctb_data/datasets/train_vis_noise_wrist_clone12.hdf5",
    #         # "ctb_data/datasets/train_vis_noise_swap_clone12.hdf5",
    #         "ctb_data/datasets/vis_wrist_clone12.hdf5",
    #         "ctb_data/datasets/noise_wrist_clone12.hdf5",
    #         "ctb_data/datasets/vis_noise_wrist_clone12.hdf5",
    #     ],
    #     value_names=[
    #         # "canonical",
    #         # "train",
    #         # "train_swap",
    #         # "train_vis",
    #         # "train_vis_noise",
    #         # "train_vis_noise_swap",
    #         "vis",
    #         "noise",
    #         "vis_noise"
    #     ]
    # )

    # generator.add_param(
    #     key="train.num_traj_clones",
    #     name="",
    #     group=0,
    #     values=[
    #         6,6,6#,6,6
    #     ],
    # )

    # # num latents experiments
    # generator.add_param(
    #     key="algo.vtt.vtt_kwargs.num_latents",
    #     name="exp",
    #     group=0,
    #     values=[
    #         1,2,4,8,16,32,64,128
    #     ],
    #     value_names=[
    #         "latents1",
    #         "latents2",
    #         "latents4",
    #         "latents8",
    #         "latents16",
    #         "latents32",
    #         "latents64",
    #         "latents128"
    #     ]
    # )

    generator.add_param(
        key="train.seed",
        name="seed",
        group=1,
        values=[
            2024#,2025,2026,2027,2028,2029
        ],
        value_names=["2024"]#,"2025","2026","2027","2028","2029"]
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

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        help="wandb project name"
    )

    args = parser.parse_args()
    main(args)
