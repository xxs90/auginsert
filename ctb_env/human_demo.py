#### FIX 
'''
    binding_utils.py (in robosuite source)
    in get_xml():
        with open(filename) as f:
                data = f.read()
'''

import numpy as np
import cv2
import robosuite as suite

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob
import gc
import traceback

import h5py
import numpy as np

import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import input2action

from robosuite.environments.base import register_env
from robosuite.devices import Keyboard
from dual_arm_keyboard import DualArmKeyboard, dualinput2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, DomainRandomizationWrapper
from ctb_data_collection_wrapper import DualArmCapTheBottleDataCollectionWrapper, CapTheBottleDataCollectionWrapper

import matplotlib.pyplot as plt

from demo_utils import DemoRecorder

from cap_the_bottle_env import CapTheBottle, CapTheBottleInitializer
from robotiq_gripper import NewRobotiq85Gripper, EVRobotiq85Gripper
from robosuite.environments.base import register_env
from robosuite.models.robots.robot_model import register_robot

register_env(CapTheBottleInitializer)

from robosuite.models.grippers import ALL_GRIPPERS, GRIPPER_MAPPING
GRIPPER_MAPPING["EVRobotiq85Gripper"] = EVRobotiq85Gripper

from robosuite.utils.transform_utils import quat2mat, mat2euler

def collect_human_trajectory(env, env_configuration, device, dual_arm=False):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()
    recorder = DemoRecorder()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    num_actions = 0
    successful = False
    while True:
        # Get the newest action
        if dual_arm:
            l_action, r_action, l_grasp, r_grasp = dualinput2action(
                device=device, robots=env.robots, env_configuration=env_configuration
            )
            if l_action is not None and r_action is not None:
                action = np.concatenate([l_action[:3], r_action[:3]]).clip(-1, 1)
            else:
                action = None
        else:
            action, grasp = input2action(
                device=device, robot=env.robots[0], active_arm="right", env_configuration=env_configuration
            )
            # TODO: Expand this to rotational actions as well
            if action is not None:
                action = action[:3].clip(-1,1)
        
        # If action is none, then this is a reset so we should break
        if action is None:
            break

        # Run environment step
        obs, reward, done, info = env.step(action)
        left_ft = obs['ft_all'][:,:6]
        right_ft = obs['ft_all'][:,6:]

        im1 = env.sim.render(height=640, width=640, camera_name="overhead")[...,::-1]
        im2 = env.sim.render(height=640, width=640, camera_name="frontview")[...,::-1]
        cv2.imshow('view', np.hstack((im1, im2))) # , im3, im4)))
        cv2.waitKey(10)
        
        # NOTE: Uncomment this if you want to visualize force-torque readings as well
        # Render demo view (rgb + ft)
        # rgbs = [obs['overhead_image'], obs['frontview_image']] # , obs['left_wristview_image']]
        # recorder.step(rgbs, left_ft, right_ft)

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            successful = True
            break

        # state machine to check for having a success for 5 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 5  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
        
        num_actions += 1

    # cleanup for end of data collection episodes
    env.close()
    return successful

def gather_demonstrations_as_hdf5(directory, out_dir, env_info, hdf5_name):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            init_perturb (attribute) - initial perturbation for demonstration (used for deterministic playback)
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, hdf5_name)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False
        init_perturb = None

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

            if init_perturb is None:
                init_perturb = dic["init_perturb"]

        if len(states) == 0:
            continue

        assert init_perturb is not None

        # Add only the successful demonstration to dataset
        if success:
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # store initital perturbation state as an attribute
            ep_data_grp.attrs["init_perturb"] = init_perturb

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")
    
    print(num_eps, "episodes saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

def main(args):
    pos_controller_config = {
        'type': 'OSC_POSE',
        'input_max': 1,
        'input_min': -1,
        'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        'kp': 150,
        'damping_ratio': 2, # 5
        'impedance_mode': 'fixed',
        'kp_limits': [0,300],
        'damping_ratio_limits': [0, 10],
        'position_limits': None,
        'orientation_limits': None,
        'uncouple_pos_ori': True,
        'control_delta': False,
        'interpolation': 'null',
        'ramp_ratio': 0.2
    }

    config = {
        "env_name": "CapTheBottleInitializer",
        "robots": ["UR5e", "UR5e"],
        "gripper_types": ["EVRobotiq85Gripper", "EVRobotiq85Gripper"],
        "controller_configs": [pos_controller_config, pos_controller_config],
        "base_pose": [0, 0, 0.625],
        "peg_perturbation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "hole_perturbation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "camera_names": ["overhead", "frontview", "left_wristview", "right_wristview"], # Demos obs (offscreen) view
        "pose_variations": ["trans"],
        "obj_variations": ["xt", "zt", "yr", "zr"],
        "obj_shape": "key",
        "obj_shape_variations": ["line", "arrow", "circle", "cross", "diamond", "hexagon", "key"],
        "peg_body_shape": "cube",
        "hole_body_shape": "cube",
        "peg_hole_swap": False,
        "obj_body_shape_variations": ["cube", "cylinder", "octagonal", "cube-thin", "cylinder-thin", "octagonal-thin"],
        "ft_noise_std": [0.0, 0.0],
        "prop_noise_std": [0.0, 0.0]
    }

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        hard_reset=True,
        ignore_done=True,
        use_camera_obs=True,
        # base_pose_perturbation=[[0,0,0,0,0,0],[0,0,0,0,0,0]],
        control_freq=20,
        perturbation_seed=20242025,
    )

    env = VisualizationWrapper(env)

    if args.dual_arm:
        device = DualArmKeyboard()
    else:
        device = Keyboard()

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    if args.record:
        # wrap the environment with data collection wrapper

        # NOTE: If you want to resume data collection for a crashed run, specify the 
        # temp folder used for previous runs in `tmp_directory`. Otherwise, uncomment
        # the line below. Make sure you use a different seed to avoid duplicated demos!
        tmp_directory = 'tmp/demo_exps' # TODO: Put desired temp directory here!
        # tmp_directory = "tmp/{}".format(str(time.time()).replace(".", "_")) # TODO: Uncomment this to start a fresh set of demos!

        if args.dual_arm:
            env = DualArmCapTheBottleDataCollectionWrapper(env, tmp_directory)
        else:
            env = CapTheBottleDataCollectionWrapper(env, tmp_directory)

        # make a new timestamped directory
        t1, t2 = str(time.time()).split(".")
        new_dir = 'ctb_data/datasets'
        os.makedirs(new_dir, exist_ok=True)
        # new_dir = os.path.join('out', "{}_{}_{}".format(config["env_name"], t1, t2))
        os.makedirs(new_dir)

    try:
        num_eps = 0
        while True:
            success = collect_human_trajectory(env, "default", device, dual_arm=args.dual_arm)
            if success:
                num_eps += 1
            else:
                print("Episode excluded from dataset")
            print(num_eps, " successful episodes collected")
            gc.collect()
            if args.record:
                gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, args.name)

    except:
        print("Collection interrupted")
        traceback.print_exc()
    finally:
        print("Collection complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', default=False, help='Records human trajectories to hdf5 file')
    parser.add_argument('--dual_arm', action='store_true', default=False, help='Record demonstrations with a dual-arm action space')
    parser.add_argument('--name', type=str, default='demo.hdf5', help='Name of the hdf5 file (if recording demonstrations)')
    args = parser.parse_args()
    main(args)