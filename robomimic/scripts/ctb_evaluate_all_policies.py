from glob import glob
import argparse
import os

# Bash script that contains all experiments in a given experiment name
# and generates commands to run ctb_evaluate_agent on all of the trained
# models. Optionally has an argument to split up sh file into multiple
# for parallel processing.

def script_from_policy_paths(args):
    """
    Generates a bash script to run the rollouts that correspond to
    the input policy paths.
    """
    policy_paths = set([p.split('/')[-1].split('_seed_')[0] for p in glob(f'experiments/{args.exp}/*')])
    policies_per_split = len(policy_paths) // args.split

    # Generate add on arguments
    add_ons = ""
    add_ons += f'--exp {args.exp} '
    add_ons += f'--n_rollouts {args.n_rollouts} '

    if args.horizon is not None:
        add_ons += f'--horizon {args.horizon} '

    if args.video_path_folder is not None:
        add_ons += f'--video_path_folder {args.video_path_folder} '
    
    add_ons += f'--video_skip {args.video_skip} '
    add_ons += f'--camera_names {arg_list_to_str(args.camera_names)} '

    if args.train_vars:
        add_ons += '--train_vars '
    
    if args.eval_vars:
        add_ons += '--eval_vars '

    if args.eval_no_swap:
        add_ons += '--eval_no_swap '

    if args.obj_shape_vars is not None:
        add_ons += f'--obj_shape_vars {arg_list_to_str(args.obj_shape_vars)} '
    
    if args.obj_body_shape_vars is not None:
        add_ons += f'--obj_body_shape_vars {arg_list_to_str(args.obj_body_shape_vars)} '
    
    if args.obj_vars is not None:
        add_ons += f'--obj_vars {arg_list_to_str(args.obj_vars)} '

    add_ons += f'--pose_vars {arg_list_to_str(args.pose_vars)} '

    if args.visual_vars is not None:
        add_ons += f'--visual_vars {arg_list_to_str(args.visual_vars)} '

    if args.var_base_pose:
        add_ons += f'--var_base_pose '
    
    if args.var_swap:
        add_ons += f'--var_swap '
    
    if args.ft_noise_std is not None:
        add_ons += f'--ft_noise_std {arg_list_to_str(args.ft_noise_std)} '
    
    if args.prop_noise_std is not None:
        add_ons += f'--prop_noise_std {arg_list_to_str(args.prop_noise_std)} '

    if args.seed is not None:
        add_ons += f'--seed {args.seed} '
    
    if args.p_seed is not None:
        add_ons += f'--p_seed {args.p_seed} '
    
    if args.set_canonical:
        add_ons += f'--set_canonical '
    
    if args.wandb_proj_name is not None:
        add_ons += f'--wandb_proj_name {args.wandb_proj_name}'
    
    num_split = args.split
    splits = [[] for _ in range(num_split)]
    spl = 0

    for policy_path in policy_paths:
        # if 'vtail' in policy_path:
        splits[spl].append(policy_path)
        spl = (spl + 1) % num_split
    
    os.makedirs(args.video_path_folder, exist_ok=True)
    
    for n, split in enumerate(splits):
        if len(split) > 0:
            with open(os.path.join(args.video_path_folder, f'run_evals{n}.sh'), 'a') as f:
                # f.write("#!/bin/bash\n\n")
                for path in split:
                    # write python command to file
                    cmd = f"python robomimic/scripts/ctb_evaluate_agents.py --policy {path} " + add_ons + "\n"
                    print()
                    print(cmd)
                    f.write(cmd)

def arg_list_to_str(arglist):
    s = ""
    for a in arglist[:-1]:
        s += str(a) + " "
    s += str(arglist[-1])
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment name
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="overall experiment name",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="split bash file for multiprocessing"
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=50,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path_folder",
        type=str,
        default=None,
        help="(optional) render rollouts to this folder",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["overhead"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # default settings for train or evaluation variations
    parser.add_argument(
        "--train_vars",
        action='store_true',
        help="use training set variations for rollouts"
    )

    parser.add_argument(
        "--eval_vars",
        action='store_true',
        help="use evaluation set variations for rollouts"
    )

    parser.add_argument(
        "--eval_no_swap",
        action='store_true',
        help="leave out swapped peg and hole from evaluation set variations"
    )

    # task variations to evaluate over
    parser.add_argument(
        "--obj_shape_vars",
        type=str,
        nargs='+',
        default=None,
        help="if provided, add peg/hole shape variations to rollouts"
    )

    parser.add_argument(
        "--obj_body_shape_vars",
        type=str,
        nargs='+',
        default=None,
        help="if provided, add object body shape variations to rollouts"
    )

    parser.add_argument(
        "--obj_vars",
        type=str,
        nargs='+',
        default=None,
        help="if provided, add object pose variations to rollouts"
    )

    parser.add_argument(
        "--pose_vars",
        type=str,
        nargs='+',
        default=['trans'],
        help="if provided, add arm pose variations to rollouts"
    )

    parser.add_argument(
        "--visual_vars",
        type=str,
        nargs='+',
        default=None,
        help="if provided, add visual variations to rollouts"
    )

    parser.add_argument(
        "--var_base_pose",
        action='store_true',
        help="if provided, vary the base insertion pose for each rollout",
    )

    parser.add_argument(
        "--var_swap",
        action='store_true',
        help="if provided, potentially switch the peg and hole arms for each rollout",
    )

    parser.add_argument(
        "--ft_noise_std",
        type=str,
        nargs='+',
        default=None,
        help="if provided, add force-torque noise variations to rollouts"
    )

    parser.add_argument(
        "--prop_noise_std",
        type=str,
        nargs='+',
        default=None,
        help="if provided, add proprioceptive noise variations to rollouts"
    )

    parser.add_argument(
        "--set_canonical",
        action='store_true',
        help="if provided, any variations not specified will be set to canonical settings (rather than the policy's defaults)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--p_seed",
        type=int,
        default=None,
        help="(optional) set seed for perturb inits",
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None,
        help="wandb project name"
    )

    args = parser.parse_args()
    script_from_policy_paths(args)
