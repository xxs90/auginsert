import os
import json
import h5py
import argparse
import numpy as np
import imageio
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.print_color_utils as PrintColorUtils
from robomimic.envs.env_base import EnvBase

class TrajectoryCloningEnv():
    def __init__(
        self,
        n_envs,
        env_meta,
        stack_obs,
        variations,
        include_canonical_env=True,
        base_perturbation_seed=None,
        **env_make_kwargs
    ):
        self.envs = []
        self.env_viable = []
        self.variations = variations
        self.n_envs = n_envs
        self.stack_obs = stack_obs
        self.canonical = include_canonical_env

        # Set environment variations
        if 'base_pose' in self.variations.keys():
            env_meta['env_kwargs']['base_pose'] = None
        
        if 'peg_hole_swap' in self.variations.keys():
            env_meta['env_kwargs']['peg_hole_swap'] = None

        if 'obj_shape_variations' in self.variations.keys():
            env_meta['env_kwargs']['obj_shape'] = None
            env_meta['env_kwargs']['obj_shape_variations'] = self.variations['obj_shape_variations']
        
        if 'obj_body_shape_variations' in self.variations.keys():
            env_meta['env_kwargs']['peg_body_shape'] = None
            env_meta['env_kwargs']['hole_body_shape'] = None
            env_meta['env_kwargs']['obj_body_shape_variations'] = self.variations['obj_body_shape_variations']
        
        if 'obj_variations' in self.variations.keys():
            env_meta['env_kwargs']['peg_perturbation'] = None
            env_meta['env_kwargs']['hole_perturbation'] = None
            env_meta['env_kwargs']['obj_variations'] = self.variations['obj_variations']

        if 'ft_noise_std' in self.variations.keys():
            env_meta['env_kwargs']['ft_noise_std'] = self.variations['ft_noise_std']
        
        if 'prop_noise_std' in self.variations.keys():
            env_meta['env_kwargs']['prop_noise_std'] = self.variations['prop_noise_std']
        
        if 'visual_variations' in self.variations.keys():
            env_meta['env_kwargs']['visual_variations'] = self.variations['visual_variations']

        # 50 percent extra environments in case of failed demos
        extra_envs = int(n_envs * 0.5)

        # Add 1 extra env for canonical demos just in case one fails
        # TODO: TEMP
        # if extra_envs <= 0:
        #     extra_envs += 1
        for n in range(n_envs+extra_envs):
            env_meta['env_kwargs']['perturbation_seed'] = base_perturbation_seed+999*n
            env = EnvUtils.create_env_for_data_processing(
                env_meta=env_meta,
                **env_make_kwargs
            )
            self.envs.append(env)
            self.env_viable.append(True)
    
    # Set only the values that have not been specified as variations (except for arm poses)
    def set_perturbation_values_from_array(self, init_perturb):
        base_pose = init_perturb[:3]
        left_perturb, right_perturb = init_perturb[3:9], init_perturb[9:15]
        peg_perturb, hole_perturb = init_perturb[15:21], init_perturb[21:27]
        obj_shape, peg_body_shape, hole_body_shape, swap = init_perturb[27], init_perturb[28], init_perturb[29], init_perturb[30]

        init_state = {
            'base_pose': base_pose,
            'obj_variations': [peg_perturb, hole_perturb],
            'obj_shape_variations': obj_shape,
            'obj_body_shape_variations': [peg_body_shape, hole_body_shape],
            'peg_hole_swap': swap,
        }

        for k in init_state.keys():
            if k in self.variations.keys():
                if isinstance(init_state[k], list):
                    init_state[k] = [None, None]
                else:
                    init_state[k] = None
        
        for env in self.envs:
            env.env.set_perturbation_values(
                base_pose=init_state['base_pose'],
                base_pose_perturb=np.vstack((left_perturb,right_perturb)),
                peg_perturb=init_state['obj_variations'][0],
                hole_perturb=init_state['obj_variations'][1],
                obj_shape=init_state['obj_shape_variations'],
                peg_body_shape=init_state['obj_body_shape_variations'][0],
                hole_body_shape=init_state['obj_body_shape_variations'][1],
                swap=init_state['peg_hole_swap'],
            )
        
        # Treat the first environment as the canonical environment
        if self.canonical:
            self.envs[0].env.set_perturbation_values_from_array(init_perturb)

    def end_episode_success_check(self):
        # Determine viability of still viable environments at the end of the episode
        for n, env in enumerate(self.envs):
            if self.env_viable[n]:
                self.env_viable[n] = env.env.check_success_forgiving()
        return self.env_viable
    
    # Get init perturb states of all environments (for playback purposes)
    def get_perturbation_values_as_array(self):
        return np.vstack([e.env.get_perturbation_values_as_array() for e in self.envs])

    # Dummy function since we are not using rewards
    def get_reward(self):
        return self.envs[0].get_reward()
    
    def get_obs(self, obs: list = None):
        # if obs is none, get initial obs from all envs
        # else, modify obs based if self.stack_obs is True
        if obs is None:
            obs = [env.get_observation() for env in self.envs]

        if self.stack_obs:
            obs = {k: np.concatenate([np.expand_dims(ob[k],axis=0) for ob in obs],axis=0) for k in obs[0].keys()}
        
        return obs
    
    def reset(self, seed=None):
        for n, env in enumerate(self.envs):
            env.env.reset_counter = 1
            env.env.reset(seed=seed)
            self.env_viable[n] = True
    
    def render_all(self, n=None):
        if n is None:
            n = self.n_envs
        video_imgs = [env.render(mode="rgb_array", height=480, width=640, camera_name="overhead") for env in self.envs[:n]]
        return np.hstack(video_imgs)
    
    def step(self, action):
        obs_all = []
        contacts = []

        for n, env in enumerate(self.envs):
            obs, _, _, _ = env.step(action)

            # An environment is inviable if a force-torque reading exceeds maximum limit (or if env does not succeed)
            ft = obs['robot0_robot1_ft_denoised-state']
            if self.env_viable[n] and (np.max(ft[...,:3]) > 100.0 or np.max(ft[...,3:6]) > 6.0 or np.max(ft[...,6:9]) > 100.0 or np.max(ft[...,9:]) > 6.0):
                print('Max force-torque threshold reached, invalidating demo...')
                self.env_viable[n] = False
            
            # Determining contact state of observation
            left_ft = ft[:,:6]
            right_ft = ft[:,6:]

            if np.linalg.norm(left_ft[...,-1,:3]) > 10.0 and np.linalg.norm(right_ft[...,-1,:3]) > 10.0:
                contacts.append(1)
            else:
                contacts.append(0)
            
            obs_all.append(obs)
        
        return self.get_obs(obs=obs_all), contacts


def clone_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    stack_obs,
    video_writer=None,
    camera_names=None,
    camera_height=84,
    camera_width=84
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (TrajectoryCloningEnv): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    assert isinstance(env, TrajectoryCloningEnv)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    init_perturb=initial_state["init_perturb"]
    env.set_perturbation_values_from_array(init_perturb)
    env.reset()
    obs = env.get_obs()

    camera_info = get_camera_info(
        env=env.envs[0],
        camera_names=camera_names, 
        camera_height=camera_height, 
        camera_width=camera_width,
    )

    traj_len = states.shape[0]

    if stack_obs:
        traj = dict(
            obs=[],
            next_obs=[],
            rewards=[],
            dones=[],
            actions=np.array(actions),
            states=np.array(states),
            initial_state_dict=initial_state,
        )
        contacts = []

        # iteration variable @t is over "next obs" indices
        for t in tqdm(range(1, traj_len + 1)):
            next_obs, contact = env.step(actions[t-1])
        
            r = env.get_reward()
            done = int(t == traj_len)
            contacts.append(contact)

            # collect transition
            traj["obs"].append(obs)
            traj["next_obs"].append(next_obs)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            # update for next iter
            obs = deepcopy(next_obs)

            if video_writer is not None:
                video_writer.append_data(env.render_all())

    # Cloned trajectories as separate demonstrations
    else:
        traj = [
            dict(
                obs=[],
                next_obs=[],
                rewards=[],
                dones=[],
                actions=np.array(actions),
                states=np.array(states),
                initial_state_dict=initial_state,
            ) for _ in range(len(env.envs))
        ]
        contacts = [[] for _ in range(len(env.envs))]

        # iteration variable @t is over "next obs" indices
        for t in tqdm(range(1, traj_len + 1)):
            next_obs, contact = env.step(actions[t-1])

            r = env.get_reward()
            done = int(t == traj_len)
            
            # collect transition
            for ob, n_ob, t, c, con in zip(obs, next_obs, traj, contact, contacts):
                t["obs"].append(ob)
                t["next_obs"].append(n_ob)
                t["rewards"].append(r)
                t["dones"].append(done)
                con.append(c)

            # update for next iter
            obs = deepcopy(next_obs)

    # Check for viable trajectory clones
    env_viable = env.end_episode_success_check()
    num_viable = sum(env_viable)

    # print("[DEBUG] ENV VIABLE:", env_viable)

    # Invalidate trajectory if canonical demo fails
    if env.canonical and not env_viable[0]:
        print(f'{PrintColorUtils.WARNING}[WARNING] Canonical environment is inviable{PrintColorUtils.ENDC}, skipping demo...')
        return None, None, None

    # Filter out inviable demos (or demos over n_env limit)
    print(f'[DEBUG] {num_viable}/{len(env_viable)} environments were viable{PrintColorUtils.ENDC}')
    if num_viable < env.n_envs:
        print(f'{PrintColorUtils.WARNING}[WARNING] Not enough environments were viable{PrintColorUtils.ENDC}, skipping demo...')
        return None, None, None
    
    # Filter out viable envs to only desired amount
    elif num_viable > env.n_envs:
        n = len(env_viable) - 1
        while num_viable > env.n_envs:
            env_viable[n] = False
            n -= 1
            num_viable = sum(env_viable)
    
    # print('[DEBUG] ENV VIABLE AFTER', env_viable)
    assert sum(env_viable) == env.n_envs, f'Got {sum(env_viable)}, expected {env.n_envs}'
    
    if stack_obs:
        # Filter out obs, next_obs, and contacts based on chosen viable environments
        for i in range(len(traj['obs'])):
            for k in traj['obs'][i].keys():
                traj['obs'][i][k] = traj['obs'][i][k][env_viable,...]
                n_test = traj['obs'][i][k].shape[0]
                assert n_test == env.n_envs, f'Got {n_test} envs for key {k}, expected {envs.n_envs}'
            for k in traj['next_obs'][i].keys():
                traj['next_obs'][i][k] = traj['next_obs'][i][k][env_viable,...]
                n_test = traj['next_obs'][i][k].shape[0]
                assert n_test == env.n_envs, f'Got {n_test} envs for key {k}, expected {envs.n_envs}'

        traj['contacts'] = np.array(contacts)[...,env_viable]
        n_test = traj['contacts'].shape[-1]
        assert n_test == env.n_envs, f'Got {n_test} envs for contacts, expected {env.n_envs}'

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
    else:
        # Filter out trajectories based on chosen viable environments
        viable_trajs = []

        for t, c, viable in zip(traj, contacts, env_viable):
            if viable:
                t['contacts'] = np.array(c)

                # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
                t["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(t["obs"])
                t["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(t["next_obs"])

                # list to numpy array
                for k in t:
                    if k == "initial_state_dict":
                        continue
                    if isinstance(t[k], dict):
                        for kp in t[k]:
                            t[k][kp] = np.array(t[k][kp])
                    else:
                        t[k] = np.array(t[k])
                viable_trajs.append(t)
        
        traj = viable_trajs
        assert len(traj) == env.n_envs, f'Got {len(traj)}, expected {env.n_envs}'

    # Store initial variation settings of recorded demos
    init_perturbs = env.get_perturbation_values_as_array()
    init_perturbs = init_perturbs[env_viable,...]

    return traj, camera_info, init_perturbs

def get_camera_info(
    env,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    assert EnvUtils.is_robosuite_env(env=env)

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
        R = env.get_camera_extrinsic_matrix(camera_name=cam_name) # camera pose in world frame
        if "wristview" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            if "left" in cam_name:
                eef_site_name = env.base_env.robots[0].controller.eef_name
            elif "right" in cam_name:
                eef_site_name = env.base_env.robots[1].controller.eef_name
            eef_pos = np.array(env.base_env.sim.data.site_xpos[env.base_env.sim.model.site_name2id(eef_site_name)])
            eef_rot = np.array(env.base_env.sim.data.site_xmat[env.base_env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info
    

def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"
    
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)

    # Override history length of force-torque if provided
    if args.ft_history_length is not None:
        env_meta['env_kwargs']['force_torque_hist_len'] = args.ft_history_length
    
    # Set variations for cloned demonstrations
    variations = {}

    if args.train_vars:
        variations['obj_shape_variations'] = ['key', 'cross', 'circle']
        variations['obj_body_shape_variations'] = ['cube', 'cylinder']
        variations['obj_variations'] = ['xt', 'zt', 'zr']
    else:
        if args.var_base_pose:
            variations['base_pose'] = None
        
        if args.var_swap:
            variations['peg_hole_swap'] = None
        
        if args.obj_shape_vars is not None:
            variations['obj_shape_variations'] = args.obj_shape_vars
        
        if args.obj_body_shape_vars is not None:
            variations['obj_body_shape_variations'] = args.obj_body_shape_vars
        
        if args.obj_vars is not None:
            variations['obj_variations'] = args.obj_vars
        
        if args.ft_noise_std is not None:
            assert len(args.ft_noise_std) == 2
            variations['ft_noise_std'] = args.ft_noise_std
        
        if args.prop_noise_std is not None:
            assert len(args.prop_noise_std) == 2
            variations['prop_noise_std'] = args.prop_noise_std
        
        if args.visual_vars is not None:
            variations['visual_variations'] = args.visual_vars
    
    # Create env for data processing
    env = TrajectoryCloningEnv(
        n_envs=args.n_envs,
        env_meta=env_meta,
        variations=variations,
        stack_obs=args.stack_obs,
        include_canonical_env=args.canonical_env,
        base_perturbation_seed=args.p_seed,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
        use_depth_obs=args.depth
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.envs[0].serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]
    
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    train_keys = []
    valid_keys = []

    total_samples = 0
    total_demos = 0

    video_writer = imageio.get_writer('vis/vis_test.mp4', fps=20)
    for ind in range(len(demos)):
        print(f'{PrintColorUtils.OKCYAN}[INFO] Recording demo {ind}{PrintColorUtils.ENDC}')
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        init_perturb = f["data/{}".format(ep)].attrs["init_perturb"]

        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["init_perturb"] = init_perturb

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj, camera_info, init_perturbs = clone_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
            stack_obs=args.stack_obs,
            video_writer=video_writer,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
        )

        # A failed trajectory clone will not be collected 
        if traj is None:
            print('[DEBUG] Restarting demo collection...')
            continue

        # If not stacked, store each trajectory-cloned episode as separate
        if args.stack_obs:
            if np.array(ep,dtype='S') in f["mask"]["valid"]:
                valid_keys.append(ep)
            else:
                train_keys.append(ep)

            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("contacts", data=np.array(traj["contacts"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if args.compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
                ep_data_grp.attrs["init_perturb"] = init_perturb # initial perturbation config for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode

            if camera_info is not None:
                assert is_robosuite_env
                ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)

            total_samples += traj["actions"].shape[0]
        else:
            for n, t in enumerate(traj):
                ep_t = ep + '00' + str(n)

                if np.array(ep,dtype='S') in f["mask"]["valid"]:
                    valid_keys.append(ep_t)
                else:
                    train_keys.append(ep_t)

                ep_data_grp = data_grp.create_group(ep_t)
                ep_data_grp.create_dataset("actions", data=np.array(t["actions"]))
                ep_data_grp.create_dataset("contacts", data=np.array(t["contacts"]))
                ep_data_grp.create_dataset("states", data=np.array(t["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(t["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(t["dones"]))
                for k in t["obs"]:
                    if args.compress:
                        ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(t["obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(t["obs"][k]))
                    if not args.exclude_next_obs:
                        if args.compress:
                            ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(t["next_obs"][k]), compression="gzip")
                        else:
                            ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(t["next_obs"][k]))
                
                # episode metadata
                if is_robosuite_env:
                    ep_data_grp.attrs["model_file"] = t["initial_state_dict"]["model"]
                    ep_data_grp.attrs["init_perturb"] = init_perturb
                ep_data_grp.attrs["num_samples"] = t["actions"].shape[0]

                if camera_info is not None:
                    assert is_robosuite_env
                    ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)
                
                total_samples += t["actions"].shape[0]
        total_demos += 1

    # copy over all filter keys that exist in the original hdf5
    train_k = "mask/train"
    valid_k = "mask/valid"
    f_out[train_k] = np.array(train_keys, dtype='S')
    f_out[valid_k] = np.array(valid_keys, dtype='S')
    
    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.envs[0].serialize(), indent=4)
    print(f"Cloned {total_demos}/{len(demos)} trajectories to {output_path}")

    f.close()
    f_out.close()
    video_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )

    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # number of clones per trajectory
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="(optional) stop after n trajectories are processed",
    )

    # whether or not to stack cloned observations (for augmentation-invariant learning)
    parser.add_argument(
        "--stack_obs",
        action='store_true',
        help="stack observations of cloned demos"
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    parser.add_argument(
        "--ft_history_length",
        type=int,
        default=None,
        help="(optional) override force-torque history length"
    )

    # flag for including depth observations per camera
    parser.add_argument(
        "--depth", 
        action='store_true',
        help="(optional) use depth observations for each camera",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    # automatically clone with training set task variations
    parser.add_argument(
        '--train_vars',
        action='store_true',
        help="(optional) clone trajectories with training set variations"
    )

    # otherwise, specify task variations to evaluate over
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

    # for seeding before starting rollouts
    parser.add_argument(
        "--p_seed",
        type=int,
        default=None,
        help="(optional) set seed for perturb inits",
    )

    parser.add_argument(
        "--canonical_env",
        action="store_true",
        help="record original demo initialization in dataset"
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)