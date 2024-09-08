import argparse
import json
import h5py
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.demo_utils import DemoRecorder, VTTDemoRecorder
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy, VTTRolloutPolicy
from robomimic.envs.wrappers import EnvWrapper

import wandb
import robomimic.macros as Macros

def rollout(
    policy,
    horizon,
    render=False,
    video_writer=False,
    video_skip=5,
    camera_names=None
):
    policy.start_episode()

    # TODO: Reset real world env
    # Make sure to process depth / RGB

    # TODO: Get observation here
    obs = None

    for step_i in range(horizon):
        act = policy(ob=obs)
        
        # Unprocess action
        act_rl = act / 1000.0

        # TODO: Provide action to real world setup
        # TODO: Update obs with new obs

        # TODO: Provide early stopping for success?
        success = False
        if success:
            break
    
    return success

def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path_folder is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # retrieving best performing model based on evaluation rollouts during training
    exp_name = args.exp
    model_paths = glob(os.path.join('experiments', exp_name, policy_params, '*', 'models', '*.pth'))
    epochs = [float(p.split('_')[-1].split('.pth')[0]) for p in model_paths]
    best_pol_idx = max(enumerate(epochs), key=lambda x: x[1])[0]
    ckpt_path = model_paths[best_pol_idx]
    print('TESTING', ckpt_path)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    config, ckpt_dict = FileUtils.config_from_checkpoint(ckpt_path=ckpt_path, verbose=False)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    num_success = 0
    num_rollouts = 0

    for i in range(args.n_rollouts):
        success = rollout(
            policy=policy,
            horizon=horizon
        )

        if success:
            num_success += 1
        num_rollouts += 1
    
    print(f'[INFO] {num_success} / {num_rollouts} successes ({num_success/num_rollouts})')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment name
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="overall experiment name",
    )

    # Path to trained model
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="path to policy folder containing saved model/*.pth files",
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

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path_folder",
        type=str,
        default=None,
        help="(optional) render rollouts to this folder",
    )

    args = parser.parse_args()
    run_trained_agent(args)