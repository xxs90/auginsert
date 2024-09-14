import os
import json
import h5py
import argparse
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.vis_utils import depth_to_rgb
from robomimic.envs.env_base import EnvBase, EnvType

import warnings
warnings.filterwarnings("ignore")


def get_force_plot(forces):
    fig, ax_f = plt.subplots(figsize=(6.4, 4.8))
    plt.title("Left Arm Forces")
    ax_f.set_ylim(-100, 100)
    ax_f.plot(np.arange(len(forces)), [x[0] for x in forces], linestyle='-', marker=".", markersize=1, color="r", label="force-x")
    ax_f.plot(np.arange(len(forces)), [x[1] for x in forces], linestyle='-', marker=".", markersize=1, color="g", label="force-y")
    ax_f.plot(np.arange(len(forces)), [x[2] for x in forces], linestyle='-', marker=".", markersize=1, color="b", label="force-z")
    ax_f.legend(loc="upper right")

    ax_f.set_ylabel('force (N)')

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.reshape((480,640,3))

    plt.close()

    return data

def get_torque_plot(torques):
    fig, ax_f = plt.subplots(figsize=(6.4, 4.8))
    plt.title("Left Arm Torques")
    ax_f.set_ylim(-2, 2)
    ax_f.plot(np.arange(len(torques)), [x[0] for x in torques], linestyle='-', marker=".", markersize=1, color="r", label="torque-x")
    ax_f.plot(np.arange(len(torques)), [x[1] for x in torques], linestyle='-', marker=".", markersize=1, color="g", label="torque-y")
    ax_f.plot(np.arange(len(torques)), [x[2] for x in torques], linestyle='-', marker=".", markersize=1, color="b", label="torque-z")
    ax_f.legend(loc="upper right")
    ax_f.set_ylabel('torque (N*m)')

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.reshape((480,640,3))

    plt.close()

    return data

def get_act_plot(act):
    fig, ax_act = plt.subplots(figsize=(6.4, 4.8))
    ax_act.set_ylim(-1, 1)
    ax_act.set_xticks(np.arange(act.shape[-1]))
    ax_act.set_xticklabels(["x", "y", "z"][:act.shape[-1]])
    ax_act.bar(np.arange(act.shape[-1]), act)

    ax_act.set_ylabel('output')
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.reshape((480,640,3))

    plt.close()

    return data

def playback_traj_and_clones(
    traj_grp,
    video_writer,
    num_stack=1,
    video_skip=5,
    image_names=None,
    depth_names=None
):
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    if depth_names is not None:
        # compute min and max depth value across trajectory for normalization
        depth_min = { k : traj_grp[f"obs/{k}"][:].min() for k in depth_names }
        depth_max = { k : traj_grp[f"obs/{k}"][:].max() for k in depth_names }
    
    num_envs = traj_grp[f"obs/{image_names[0]}"][:].shape[1]
    if num_stack is None:
        num_stack = num_envs

    print(f'[DEBUG]: {num_envs} environments')
    
    traj_len = traj_grp["actions"].shape[0]
    for i in tqdm(range(traj_len)):
        all_frames = []
        if video_count % video_skip == 0:
            for n in range(num_stack):
                # Record images
                im = [traj_grp[f"obs/{k}"][i,n] for k in image_names]
                im = [cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR) for img in im]

                # Record depths
                depth = [depth_to_rgb(traj_grp[f"obs/{k}"][i,n], depth_min=depth_min[k], depth_max=depth_max[k]) for k in depth_names] if depth_names is not None else []
                depth = [cv2.resize(d, (480, 480), interpolation=cv2.INTER_LINEAR) for d in depth]

                # Record (left arm) forces
                force = get_force_plot(traj_grp[f"obs/robot0_robot1_forcetorque-state"][i,n,:,:3])

                # Record (left arm) torques
                torque = get_torque_plot(traj_grp[f"obs/robot0_robot1_forcetorque-state"][i,n,:,3:6])

                # Record actions
                # print(np.max(traj_grp["actions"][:][i]))
                act = get_act_plot(traj_grp["actions"][:][i])

                frame_all = np.hstack(im + depth + [force,torque,act])
                all_frames.append(frame_all)
            
            stacked_frame = np.vstack(all_frames)
            video_writer.append_data(stacked_frame)
        video_count += 1

def playback_dataset(args):
    assert args.video_folder is not None

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n_demos is not None:
        demos = demos[:args.n_demos]
    
    video_writer = imageio.get_writer(os.path.join(args.video_folder, args.dataset.split('/')[-1].split('.')[0]+'.mp4'), fps=20)
    for ind in range(len(demos)):
        ep = demos[ind]
        print(f"Playing back episode: {ep}")

        playback_traj_and_clones(
            traj_grp=f[f"data/{ep}"],
            video_writer=video_writer,
            num_stack=args.n_stack,
            video_skip=args.video_skip,
            image_names=args.render_image_names,
            depth_names=args.render_depth_names
        )
    
    f.close()
    video_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n_demos",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # number of clones to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n_stack",
        type=int,
        default=None,
        help="(optional) playback n clones per trajectory",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_folder",
        type=str,
        default="vis",
        help="(optional) render trajectories to this directory",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # depth observations to use for writing to video
    parser.add_argument(
        "--render_depth_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) depth observation(s) to use for rendering to video"
    )

    args = parser.parse_args()
    playback_dataset(args)