import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.print_color_utils as PrintColorUtils
from robomimic.envs.env_base import EnvBase

class DataCollectionEnv():
    def __init__(
        self,
        n_envs,
        env_meta,
        variations,
        include_canonical_env=True,
        base_perturbation_seed=None,
        **env_make_kwargs
    ):
        self.variations = variations
        self.n_envs = n_envs
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
        
        # Initialize environment
        env_meta['env_kwargs']['perturbation_seed'] = base_perturbation_seed
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            **env_make_kwargs
        )

    # Canonical setup, also resets noise and deactivates visual variations
    def set_perturbation_values_from_array(self, init_perturb):
        self.env.env.set_perturbation_values_from_array(init_perturb)
        self.env.env.ft_noise_std = np.array([0.0, 0.0], dtype=np.float64)
        self.env.env.prop_noise_std = np.array([0.0, 0.0], dtype=np.float64)
        self.env.env.arena_texture = 'light-wood'
        if 'arena_texture' in self.env.env.randomized_values:
            self.env.env.randomized_values.remove('arena_texture')
        self.env.env.visual_vars_active = False
        
    
    # Canonical env locks variations, want to unlock them for trajectory clones
    # Call this function AFTER setting variations
    def unlock_variation_generators(self):
        randomized_vals = self.env.env.randomized_values
        if 'base_pose' in self.variations.keys() and 'base_pose' not in randomized_vals:
            self.env.env.randomized_values.append('base_pose')
        
        if 'peg_hole_swap' in self.variations.keys() and 'peg_hole_swap' not in randomized_vals:
            self.env.env.randomized_values.append('peg_hole_swap')

        if 'obj_shape_variations' in self.variations.keys() and 'obj_shape' not in randomized_vals:
            self.env.env.randomized_values.append('obj_shape')
        
        if 'obj_body_shape_variations' in self.variations.keys():
            if 'peg_body_shape' not in randomized_vals:
                self.env.env.randomized_values.append('peg_body_shape')
            if 'hole_body_shape' not in randomized_vals:
                self.env.env.randomized_values.append('hole_body_shape')

        if 'obj_variations' in self.variations.keys():
            if 'peg_perturb' not in randomized_vals:
                self.env.env.randomized_values.append('peg_perturb')
            if 'hole_perturb' not in randomized_vals:
                self.env.env.randomized_values.append('hole_perturb')
        
        if 'ft_noise_std' in self.variations.keys():
            self.env.env.ft_noise_std = np.array(self.variations['ft_noise_std'], dtype=np.float64)
        
        if 'prop_noise_std' in self.variations.keys():
            self.env.env.prop_noise_std = np.array(self.variations['prop_noise_std'], dtype=np.float64)
        
        if 'visual_variations' in self.variations.keys():
            self.env.env.visual_vars_active = True
            if 'arena-train' or 'arena-eval' in self.variations['visual_variations']:
                self.env.env.randomized_values.append('arena_texture')
    
    # Dummy function since we are not using rewards
    def get_reward(self):
        return self.env.get_reward()
    
    def reset(self, seed=None):
        self.env.env.reset_counter = 1
        self.env.env.reset(seed=seed)
    
    def check_success(self):
        return self.env.env.check_success_forgiving()
    
    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        ft = obs['robot0_robot1_ft_denoised-state']

        # An environment is inviable if a force-torque reading exceeds maximum limit (or if env does not succeed)
        valid = True
        if np.max(ft[...,:3]) > 100.0 or np.max(ft[...,3:6]) > 6.0 or np.max(ft[...,6:9]) > 100.0 or np.max(ft[...,9:]) > 6.0:
            print(f'{PrintColorUtils.WARNING}[WARNING] Maximum force-torque threshold reached, skipping demo...{PrintColorUtils.ENDC}')
            valid = False
        
        return obs, valid

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

def clone_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    camera_names=None,
    camera_height=84,
    camera_width=84
):
    assert isinstance(env, DataCollectionEnv)

    # get the initial state
    init_perturb = initial_state["init_perturb"]

    camera_info = get_camera_info(
        env=env.env,
        camera_names=camera_names, 
        camera_height=camera_height, 
        camera_width=camera_width,
    )

    traj_len = states.shape[0]

    # maximum number of times to run through an environment
    max_envs = int((env.n_envs + 0.5 * env.n_envs)) + 1
    trajs = []

    for i in range(max_envs):
        if env.canonical and i == 0:
            print(f'{PrintColorUtils.OKCYAN}[INFO] Recording canonical demo{PrintColorUtils.ENDC}')
        else:
            print(f'{PrintColorUtils.OKCYAN}[INFO] Recording clone {len(trajs)}{PrintColorUtils.ENDC}')

        # load the initial state
        env.set_perturbation_values_from_array(init_perturb)

        # reactivate variations for non-canonical envs
        if not env.canonical or i > 0:
            env.unlock_variation_generators()
        
        env.reset()
        obs, _ = env.step(np.zeros(3))
        
        traj = dict(
            obs=[],
            next_obs=[],
            rewards=[],
            dones=[]
        )

        # iteration variable @t is over "next obs" indices
        valid_demo = True
        for t in tqdm(range(1, traj_len+1)):
            next_obs, valid = env.step(actions[t-1].copy())
            valid_demo = valid_demo and valid

            if not valid:
                break
            
            r = env.get_reward()
            done = int(t == traj_len)

            # collect transition
            traj["obs"].append(obs)
            traj["next_obs"].append(next_obs)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            # update for next iter
            obs = deepcopy(next_obs)
        
        valid_demo = valid_demo and env.check_success()
        if valid_demo:
            trajs.append(traj)
        elif env.canonical and i == 0:
            print(f'{PrintColorUtils.WARNING}[WARNING] Canonical environment is inviable{PrintColorUtils.ENDC}, skipping demo...')
            return None, None, None
        
        if len(trajs) == env.n_envs:
            # Stack observations for all trajectories
            traj_all = dict(
                obs=[],
                next_obs=[],
                rewards=[],
                dones=[],
                actions=np.array(actions),
                states=np.array(states),
                initial_state_dict=initial_state
            )
            for ob in ['obs', 'next_obs']:
                for step in range(traj_len):
                    all_step_obs = [traj[ob][step] for traj in trajs]
                    traj_all[ob].append(
                        {k: np.concatenate([np.expand_dims(ob_step[k], axis=0) for ob_step in all_step_obs],axis=0) for k in all_step_obs[0].keys()}
                    )
            traj_all['rewards'] = trajs[0]['rewards']
            traj_all['dones'] = trajs[0]['dones']

            # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
            traj_all["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj_all["obs"])
            traj_all["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj_all["next_obs"])

            # list to numpy array
            for k in traj_all.keys():
                if k == "initial_state_dict":
                    continue
                if isinstance(traj_all[k], dict):
                    for kp in traj_all[k]:
                        traj_all[k][kp] = np.array(traj_all[k][kp])
                else:
                    traj_all[k] = np.array(traj_all[k])
            
            return traj_all, camera_info, init_perturb

    print(f'{PrintColorUtils.WARNING}[WARNING] Not enough environments were viable{PrintColorUtils.ENDC}, skipping demo...')
    return None, None, None

def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"
    
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)

    # Override history length of force-torque if provided
    if args.ft_history_length is not None:
        env_meta['env_kwargs']['force_torque_hist_len'] = args.ft_history_length
    
    # Set variations for cloned demonstrations
    variations = {}
    
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
    env = DataCollectionEnv(
        n_envs=args.n_envs,
        env_meta=env_meta,
        variations=variations,
        include_canonical_env=args.canonical_env,
        base_perturbation_seed=args.p_seed,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
        use_depth_obs=args.depth
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.env.serialize(), indent=4))
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
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
        )

        # A failed trajectory clone will not be collected 
        if traj is None:
            print('[DEBUG] Restarting demo collection...')
            continue
        
        if np.array(ep,dtype='S') in f["mask"]["valid"]:
            valid_keys.append(ep)
        else:
            train_keys.append(ep)

        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
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
        total_demos += 1

    # copy over all filter keys that exist in the original hdf5
    train_k = "mask/train"
    valid_k = "mask/valid"
    f_out[train_k] = np.array(train_keys, dtype='S')
    f_out[valid_k] = np.array(valid_keys, dtype='S')
    
    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.env.serialize(), indent=4)
    print(f"Cloned {total_demos}/{len(demos)} trajectories to {output_path}")

    f.close()
    f_out.close()

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