"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np

from demo_utils import DemoRecorder
from robotiq_gripper import NewRobotiq85Gripper, EVRobotiq85Gripper

import robosuite

from cap_the_bottle_env import CapTheBottleInitializer
from robosuite.environments.base import register_env

register_env(CapTheBottleInitializer)
from robosuite.models.grippers import ALL_GRIPPERS, GRIPPER_MAPPING
GRIPPER_MAPPING["NewRobotiq85Gripper"] = NewRobotiq85Gripper
GRIPPER_MAPPING["EVRobotiq85Gripper"] = EVRobotiq85Gripper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env_info["base_pose"] = [0, 0, 0.625]

    env_info["obj_shape"] = None
    env_info["obj_shape_variations"] = ["arrow", "circle", "cross", "diamond", "hexagon", "key", "pentagon", "line", "u"]

    # Setting variation parameters for demo initializations
    env_info["peg_perturbation"] = None
    env_info["hole_perturbation"] = None
    env_info["obj_variations"] = ["xt", "zt", "yr"]
    env_info["camera_names"] = ["overhead", "frontview", "left_wristview", "right_wristview"]

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        # pose_variations=["trans"]
        # camera_names="overhead"
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    while True: # for ep in demos:
        ep = random.choice(demos)
        print("Playing cloned trajectories for demo", ep)

        init_perturb = f["data/{}".format(ep)].attrs["init_perturb"]
        env.set_perturbation_values(
            base_pose_perturb=np.vstack((init_perturb[3:9], init_perturb[9:15]))
        )

        for _ in range(1):

            # Little hack so that the first episode played works correctly
            # Environment only resets initial state if it has been stepped through already (prevents redundant resets)
            env.reset_counter = 1
            env.reset()

            # TODO: Why doesn't this work?
            # model_xml = f["data/{}".format(ep)].attrs["model_file"]
            # xml = env.edit_model_xml(model_xml)
            # env.reset_from_xml_string(xml)
            # env.sim.reset()

            env.viewer.set_camera(0)

            # load the flattened mujoco states
            states = f["data/{}/states".format(ep)][()]

            # load the initial state
            # env.sim.set_state_from_flattened(states[0])
            # env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]
            print("PLAYING", num_actions, "ACTIONS")

            recorder = DemoRecorder()

            for j, action in enumerate(actions):
                obs, reward, done, info = env.step(action)
                left_ft = obs['ft_all'][:,:6]
                right_ft = obs['ft_all'][:,6:]
                rgbs = [obs['overhead_image'], obs['frontview_image'], obs['left_wristview_image']]
                # env.render()

                # Visualize actions rather than torques
                # left_ft[-1, 3:] = action
                recorder.step(rgbs, left_ft, right_ft)

            print("Orig success:", env._check_success())
            print("Relaxed success:", env.check_success_forgiving())

                # if j < num_actions - 1:
                #     # ensure that the actions deterministically lead to the same recorded states
                #     state_playback = env.sim.get_state().flatten()
                #     if not np.all(np.equal(states[j + 1], state_playback)):
                #         err = np.linalg.norm(states[j + 1] - state_playback)
                #         print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

    f.close()