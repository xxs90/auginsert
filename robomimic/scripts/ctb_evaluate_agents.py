"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

    p_seed (int): if provided, set seed for perturbation inits

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
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


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy) or isinstance(policy, VTTRolloutPolicy)
    assert not (render and (video_writer is not None))

    # Create attention visualizations for VTT policies
    vtt_policy = isinstance(policy, VTTRolloutPolicy)

    policy.start_episode()
    obs = env.reset()

    # Deal with environment wrappers
    wrapped_env = env
    if isinstance(env, EnvWrapper):
        unwrapped_env = env.unwrapped
    else:
        unwrapped_env = env

    state_dict = unwrapped_env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    # obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    if vtt_policy:
        recorder = VTTDemoRecorder()
    else:
        recorder = DemoRecorder()

    try:
        for step_i in range(horizon):

            # get action from policy (and attns if vtt policy)
            if vtt_policy:
                act, img_attn, tactile_attn, proportions = policy(ob=obs)
                img_attn = [ia.detach().cpu().numpy() for ia in img_attn]
                tactile_attn = tactile_attn.detach().cpu().numpy()
                proportions = [x.detach().cpu().numpy() for x in proportions]
            else:
                act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = wrapped_env.step(act)

            print('='*20)
            for k in next_obs.keys():
                print(k, ':', next_obs[k].shape)

            # NOTE: Frame stacking adds another dimension to the observations (good for rgb history, annoying for everything else...)
            ft = next_obs['robot0_robot1_forcetorque-state']
            if len(ft.shape) == 2:
                left_ft = ft[:,:6]
                right_ft = ft[:,6:]
            else:
                left_ft = ft[-1,:,:6]
                right_ft = ft[-1,:,6:]
            # # Hijack torque reading for action visualization
            # left_ft[-1,3:] = act

            # Use denoised force-torque for failing on threshold
            ft = next_obs['robot0_robot1_ft_denoised-state']

            # If force/torque readings are above maximum thresholds (after a few steps), fail the demo early
            if np.max(ft[...,:3]) > 100.0 or np.max(ft[...,3:6]) > 6.0 or np.max(ft[...,6:9]) > 100.0 or np.max(ft[...,9:]) > 6.0:
                if step_i >= 10:
                    print('[EVAL] Maximum force-torque threshold passed, failing demo...')
                    done = True

            # compute reward
            total_reward += r
            success = unwrapped_env.is_success()["task"]

            # visualization
            if render:
                unwrapped_env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    # Visualize attention for VTT policies
                    if vtt_policy:
                        # Create attention plots
                        cmap = plt.get_cmap('jet')
                        heatmaps = [(cmap(ia.squeeze())[...,:-1] * 255.0).astype(np.uint8) for ia in img_attn]
                        proportion_plot = recorder.get_attn_proportion_plot(proportions)

                        video_img = []
                        for cam_name in camera_names:
                            video_img.append(env.render(mode="rgb_array", height=480, width=640, camera_name=cam_name))
                        
                        # Add heatmap on top of rgb observations
                        # overhead = (obs['overhead_image'].transpose((1,2,0)) * 255.0).astype(np.uint8)
                        l_wristview = (obs['left_wristview_image'].transpose((1,2,0)) * 255.0).astype(np.uint8)
                        r_wristview = (obs['right_wristview_image'].transpose((1,2,0)) * 255.0).astype(np.uint8)
                        super_imposed_imgs = [
                            cv2.addWeighted(heatmap, 0.5, a, 0.5, 0) for heatmap, a in zip(heatmaps, [l_wristview, r_wristview])
                        ]
                        front_video_img = env.render(mode="rgb_array", camera_name="frontview", height=480, width=640)
                        video_imgs = [cv2.resize(sp_img, (480, 480), interpolation = cv2.INTER_LINEAR) for sp_img in super_imposed_imgs]
                        hist_len = policy.policy.nets["policy"].nets["encoder"].nets["obs"].obs_nets['vtt'].tactile_history
                        attn_plot = VTTDemoRecorder.get_force_attn_plot(left_ft, tactile_attn, hist_len=hist_len)
                        video_writer.append_data(np.hstack((front_video_img, *video_imgs, attn_plot, proportion_plot)))
                    else:
                        video_img = []
                        for cam_name in camera_names:
                            video_img.append(unwrapped_env.render(mode="rgb_array", height=480, width=640, camera_name=cam_name))
                        all_frames = recorder.step(video_img, left_ft, right_ft, act)
                        video_writer.append_data(all_frames)

                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success or step_i == horizon-1:
                if vtt_policy:
                    # Create attention plots
                    cmap = plt.get_cmap('jet')
                    heatmaps = [(cmap(ia.squeeze())[...,:-1] * 255.0).astype(np.uint8) for ia in img_attn]
                    proportion_plot = recorder.get_attn_proportion_plot(proportions)

                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=480, width=640, camera_name=cam_name))
                    
                    # Add heatmap on top of rgb observations
                    # overhead = (obs['overhead_image'].transpose((1,2,0)) * 255.0).astype(np.uint8)
                    l_wristview = (obs['left_wristview_image'].transpose((1,2,0)) * 255.0).astype(np.uint8)
                    r_wristview = (obs['right_wristview_image'].transpose((1,2,0)) * 255.0).astype(np.uint8)
                    super_imposed_imgs = [
                        cv2.addWeighted(heatmap, 0.5, a, 0.5, 0) for heatmap, a in zip(heatmaps, [l_wristview, r_wristview])
                    ]
                    front_video_img = env.render(mode="rgb_array", camera_name="frontview", height=480, width=640)
                    video_imgs = [cv2.resize(sp_img, (480, 480), interpolation = cv2.INTER_LINEAR) for sp_img in super_imposed_imgs]
                    hist_len = policy.policy.nets["policy"].nets["encoder"].nets["obs"].obs_nets['vtt'].tactile_history
                    attn_plot = VTTDemoRecorder.get_force_attn_plot(left_ft, tactile_attn, hist_len=hist_len)
                    all_frames = np.hstack((front_video_img, *video_imgs, attn_plot, proportion_plot))
                else:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(unwrapped_env.render(mode="rgb_array", height=480, width=640, camera_name=cam_name))
                    all_frames = recorder.step(video_img, left_ft, right_ft, act)
                mask = np.zeros(all_frames.shape, dtype=np.uint8)
                if success:
                    mask[...,1] = 255
                else:
                    mask[...,0] = 255
                all_frames = cv2.addWeighted(all_frames, 0.7, mask, 0.3, 0)
                for i in range(10):
                    video_writer.append_data(all_frames)
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args, seed):
    # some arg checking
    write_video = (args.video_path_folder is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # retrieving best performing model based on evaluation rollouts during training
    exp_name = args.exp
    policy_params = args.policy + '_seed_' + seed
    model_paths = glob(os.path.join('experiments', exp_name, policy_params, '*', 'models', '*success*.pth'))
    success_rates = [float(p.split('_')[-1].split('.pth')[0]) for p in model_paths]
    print(success_rates)
    best_pol_idx = max(enumerate(success_rates), key=lambda x: x[1])[0]
    ckpt_path = model_paths[best_pol_idx]
    print('TESTING', ckpt_path)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    config, ckpt_dict = FileUtils.config_from_checkpoint(ckpt_path=ckpt_path, verbose=False)

    # restore policy
    if args.visualize_attns:
        policy, ckpt_dict = FileUtils.vtt_policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    else:
    # TODO: TEMP (while using ModalityIndependentVTT)
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # Set canonical params if option is given
    if args.set_canonical or args.train_vars or args.eval_vars:
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape"] = "key"
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_body_shape"] = "cube"
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_body_shape"] = "cube"
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_perturbation"] = [0,0,0,0,0,0]
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_perturbation"] = [0,0,0,0,0,0]
        ckpt_dict["env_metadata"]["env_kwargs"]["base_pose"] = [0,0,0.625]
        ckpt_dict["env_metadata"]["env_kwargs"]["ft_noise_std"] = [0.0, 0.0]
        ckpt_dict["env_metadata"]["env_kwargs"]["prop_noise_std"] = [0.0, 0.0]
        ckpt_dict["env_metadata"]["env_kwargs"]["visual_variations"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_hole_swap"] = False

    if args.p_seed is not None:
        ckpt_dict["env_metadata"]["env_kwargs"]["perturbation_seed"] = args.p_seed
    
    if args.train_vars:
        # Grasp variations: xt, zt, zr
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_perturbation"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_perturbation"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_variations"] = ['xt', 'zt', 'zr']

        # Peg/hole shape variations: key, cross, circle
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape_variations"] = ['key', 'cross', 'circle']
        
        # Object body shape variations: cube, cylinder
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_body_shape"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_body_shape"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_body_shape_variations"] = ['cube', 'cylinder']
        
    elif args.eval_vars:
        # Grasp variations: xt, zt, zr, yr
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_perturbation"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_perturbation"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_variations"] = ['xt', 'zt', 'yr', 'zr']

        # Peg/hole shape variations: arrow, line, pentagon, hexagon, diamond, u
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape_variations"] = ['arrow', 'line', 'pentagon', 'hexagon', 'diamond', 'u']

        # Object body shape variations: octagonal, cube-thin (peg), cylinder-thin (peg), octagonal-thin (peg)
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_body_shape"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_body_shape"] = None
        ckpt_dict["env_metadata"]["env_kwargs"]["obj_body_shape_variations"] = ['cube-thin', 'cylinder-thin', 'octagonal-thin']

        # Visual variations: camera angle, lighting, object color, arena texture
        ckpt_dict["env_metadata"]["env_kwargs"]["visual_variations"] = ['camera', 'lighting', 'texture', 'arena-eval']

        # Force-torque noise
        ckpt_dict["env_metadata"]["env_kwargs"]["ft_noise_std"] = [5.0, 0.15]
        
        # Proprioception noise
        ckpt_dict["env_metadata"]["env_kwargs"]["prop_noise_std"] = [0.001, 0.01]

        # Peg/hole swap
        ckpt_dict["env_metadata"]["env_kwargs"]["peg_hole_swap"] = not args.eval_no_swap

    else:
        if args.obj_shape_vars is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["obj_shape_variations"] = args.obj_shape_vars
        
        if args.obj_body_shape_vars is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["peg_body_shape"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["hole_body_shape"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["obj_body_shape_variations"] = args.obj_body_shape_vars
        
        if args.obj_vars is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["peg_perturbation"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["hole_perturbation"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["obj_variations"] = args.obj_vars
        
        if args.pose_vars is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["left_pose_perturbation"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["right_pose_perturbation"] = None
            ckpt_dict["env_metadata"]["env_kwargs"]["pose_variations"] = args.pose_vars
        
        if args.visual_vars is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["visual_variations"] = args.visual_vars
        
        if args.var_base_pose:
            ckpt_dict["env_metadata"]["env_kwargs"]["base_pose"] = None
        
        if args.ft_noise_std is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["ft_noise_std"] = [float(f) for f in args.ft_noise_std]
        
        if args.prop_noise_std is not None:
            ckpt_dict["env_metadata"]["env_kwargs"]["prop_noise_std"] = [float(f) for f in args.prop_noise_std]
        
        if args.var_swap:
            ckpt_dict["env_metadata"]["env_kwargs"]["peg_hole_swap"] = True

    print('TESTING', ckpt_path)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=None, 
        render=args.render, 
        render_offscreen=(args.video_path_folder is not None), 
        verbose=False,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        # make video folder if it doesn't exist
        variations = []
        if args.train_vars:
            variations.append("train_vars")
        elif args.eval_vars:
            variations.append("eval_vars")
        else:
            if args.obj_vars is not None:
                variations.append('grasp_eval')
            elif args.obj_shape_vars is not None:
                variations.append('peg_hole_shape_eval')
            elif args.obj_body_shape_vars is not None:
                variations.append('obj_body_shape_eval')
            elif args.visual_vars is not None:
                if 'camera' in args.visual_vars:
                    variations.append('camera_angle')
                else:
                    variations.append('visual')
            elif args.var_swap:
                variations.append('peg_hole_swap')
            elif args.ft_noise_std is not None or args.prop_noise_std is not None:
                if args.ft_noise_std is not None:
                    if args.prop_noise_std is not None:
                        variations.append('sensor_noise')
                    else:
                        variations.append('ft_noise')
                elif args.prop_noise_std is not None:
                    variations.append('prop_noise')
            elif args.set_canonical:
                variations.append('canonical')
            else:
                variations.append('default')
        if args.visualize_attns:
            variations.append('attn_vis')
        
        vid_folder_path = os.path.join(args.video_path_folder, *variations)
        os.makedirs(vid_folder_path, exist_ok=True)
        vid_name = f'{policy_params}.mp4'
        vid_path = os.path.join(vid_folder_path, vid_name)
        video_writer = imageio.get_writer(vid_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))
    
    # # Log to WandB
    # wandb.log(avg_rollout_stats)
    return avg_rollout_stats

def setup_wandb(args):
    # set up wandb api key if specified in macros
    if Macros.WANDB_API_KEY is not None:
        os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

    assert Macros.WANDB_ENTITY is not None, "WANDB_ENTITY macro is set to None." \
            "\nSet this macro in {base_path}/macros_private.py" \
            "\nIf this file does not exist, first run python {base_path}/scripts/setup_macros.py".format(base_path=robomimic.__path__[0])
    
    tags = []

    if args.train_vars:
        tags.append("train_vars")
    elif args.eval_vars:
        tags.append("eval_vars")
        if args.eval_no_swap:
            tags.append("no_swap")
    else:
        if args.obj_vars is not None:
            tags.append('grasp_eval')
        elif args.obj_shape_vars is not None:
            tags.append('peg_hole_shape_eval')
        elif args.obj_body_shape_vars is not None:
            tags.append('obj_body_shape_eval')
        elif args.visual_vars is not None:
            if 'camera' in args.visual_vars:
                tags.append('camera_angle')
            else:
                tags.append('visual')
        elif args.var_swap:
            tags.append('peg_hole_swap')
        elif args.ft_noise_std is not None or args.prop_noise_std is not None:
            if args.ft_noise_std is not None:
                if args.prop_noise_std is not None:
                    tags.append('sensor_noise')
                else:
                    tags.append('ft_noise')
            elif args.prop_noise_std is not None:
                tags.append('prop_noise')
        elif args.set_canonical:
            tags.append('canonical')
        else:
            tags.append('default')
    
    wandb.init(
        entity=Macros.WANDB_ENTITY,
        project=args.wandb_proj_name,
        group=args.exp,
        name=args.policy,
        tags=tags,
        mode="online"
    )

def run_all_seeds(args):
    if args.wandb_proj_name is not None:
        setup_wandb(args)

    all_seeds = sorted(glob(os.path.join('experiments', args.exp, args.policy + '_seed_*')))
    all_results = dict()

    for exp in all_seeds:
        seed = exp.split('_')[-1]
        results = run_trained_agent(args, seed)
        all_results[seed] = results
    
    # Aggregate data from all runs
    seed_avg_results = avg_all_seeds(all_results)
    if args.wandb_proj_name is not None:
        wandb.log(seed_avg_results)

def avg_all_seeds(results):
    print(results[list(results.keys())[0]])
    success_rates = np.array([results[r]["Success_Rate"] for r in results])
    success_rate_mean = np.mean(success_rates)
    success_rate_std = np.std(success_rates)
    all_results = dict()
    for seed in results.keys():
        all_results["Success_" + seed] = results[seed]["Success_Rate"]
    all_results["Success_Rate_Mean"] = success_rate_mean
    all_results["Success_Rate_Std"] = success_rate_std
    return all_results

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
        default=None,
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


    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
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
        "--visualize_attns",
        action='store_true',
        help="if provided, will overlay modality-specific attn heatmaps over observations"
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None,
        help="wandb project name"
    )

    args = parser.parse_args()
    run_all_seeds(args)

