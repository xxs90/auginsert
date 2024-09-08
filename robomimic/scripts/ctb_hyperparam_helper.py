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


def make_generator(config_file, script_file, num_splits):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file, wandb_proj_name="ctb_vtail_percio_experiments", num_splits=num_splits
    )

    # Proprioception: on
    generator.add_param(
        key="observation.modalities.obs.low_dim",
        name="prop",
        group=0,
        values=[
            ["robot0_robot1_proprioception-state"]
        ],
        value_names=[
            "on"
        ]
    )

    # Force-torque: off/on
    generator.add_param(
        key="observation.modalities.obs.ft",
        name="ft",
        group=1,
        values=[
            # [],
            ["robot0_robot1_forcetorque-state"],
            # ["robot0_robot1_forcetorque-state"],
            # ["robot0_robot1_forcetorque-state"]
        ],
        value_names=[
            # "off",
            "on",
        ]
    )

    # Force-torque encoders
    generator.add_param(
        key="observation.encoder.ft.core_kwargs.encoder_type",
        name="ft_encoder",
        group=1,
        values=[
            # "",
            # "rnn",
            "causalconv",
            # "selfattn",
            # "transformer"
        ],
        value_names=[
            # "none",
            # "rnn",
            "causalconv",
            # "selfattn",
            # "transformer"
        ]
    )

    # Force-torque: Encoder keyword args
    generator.add_param(
        key="observation.encoder.ft.core_kwargs.encoder_kwargs",
        name="",
        group=1,
        values=[
            # {}, # none
            # {   # mlp
            #     "output_dim": 64,
            #     "layer_dims": [512,512]
            # },
            # {   # rnn
            #     "rnn_hidden_dim": 64,
            #     "rnn_num_layers": 2,
            #     "rnn_type": "LSTM"
            # },
            {   # causalconv
                "output_dim": 128,
                "activation": "relu",
                "out_channels": (16, 32, 64, 128),
                "kernel_size": (2, 2, 2, 2, 2),
                "stride": (2, 2, 2, 2, 2),
            },
            # {   # selfattn
            #     "seq_len": 32,
            #     "output_dim": 64,
            #     "d_model": 64
            # },
            # {   # transformer
            #     "seq_len": 32,
            #     "output_dim": 64,
            #     "d_model": 128,
            #     "num_heads": 4
            # }
        ]
    )

    # RGB
    generator.add_param(
        key="observation.modalities.obs.rgb",
        name="",
        group=2,
        values=[
            # [],
            ['overhead_image'],
            # ['overhead_image','left_wristview_image','right_wristview_image']
        ],
        # value_names=[
        #     # 'none',
        #     # 'top_only',
        #     # 'on'
        # ]
    )

    # RGB
    generator.add_param(
        key="observation.modalities.obs.depth",
        name="",
        group=2,
        values=[
            # [],
            ['overhead_depth'],
            # ['overhead_image','left_wristview_image','right_wristview_image']
        ],
        # value_names=[
        #     # 'none',
        #     # 'top_only',
        #     'on'
        # ]
    )

    # MLP policy
    generator.add_param(
        key="algo.rnn.enabled",
        name="pol", 
        group=3, 
        values=[False],
        value_names=["mlp"]
    )

    generator.add_param(
        key="train.seq_length",
        name="",
        group=3,
        values=[1]
    )

    generator.add_param(
        key="train.frame_stack",
        name="",
        group=3,
        values=[1]
    )

    generator.add_param(
        key="algo.vtt.enabled",
        name="",
        group=3,
        values=[False]
    )

    generator.add_param(
        key="train.data",
        name="train",
        group=4,
        values=[
            "ctb_data/datasets/vtail_demo_canonical_depth.hdf5",
            "ctb_data/datasets/vtail_demo_trainset_depth_unstacked4.hdf5",
            "ctb_data/datasets/vtail_demo_trainset_depth_unstacked12.hdf5",
            # "ctb_data/datasets/vtail_demo_trainset_depth_stacked12.hdf5",
        ],
        value_names=[
            "canonical",
            "tclone4",
            "tclone12"
            # "vtail"
        ]
    )

    # generator.add_param(
    #     key="algo.ail.enabled",
    #     name="",
    #     group=4,
    #     values=[False],
    # )

    return generator


def main(args):

    # make config generator
    generator = make_generator(config_file=args.config, script_file=args.script, num_splits=args.num_splits)

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
