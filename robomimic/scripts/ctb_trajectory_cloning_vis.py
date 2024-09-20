import os
import json
import h5py
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.print_color_utils as PrintColorUtils
from robomimic.envs.env_base import EnvBase

def get_force_plot(forces):
    fig, ax_f = plt.subplots(figsize=(6.4, 4.8))
    ax_f.set_ylim(-100, 100)
    ax_f.plot(np.arange(1,33), [x[0] for x in forces], linestyle='-', marker=".", markersize=1, color="r", label="force-x")
    ax_f.plot(np.arange(1,33), [x[1] for x in forces], linestyle='-', marker=".", markersize=1, color="g", label="force-y")
    ax_f.plot(np.arange(1,33), [x[2] for x in forces], linestyle='-', marker=".", markersize=1, color="b", label="force-z")
    ax_f.legend(loc="upper right")

    ax_f.set_ylabel('Force (N)')

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.reshape((480,640,3))

    plt.close()

    return data

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
            if 'arena-train' in self.variations['visual_variations'] or 'arena-eval' in self.variations['visual_variations']:
                self.env.env.randomized_values.append('arena_texture')
    
    # Dummy function since we are not using rewards
    def get_reward(self):
        return self.env.get_reward()
    
    def reset(self, seed=None):
        self.env.env.reset_counter = 1
        self.env.env.reset(seed=seed)
    
    def check_success(self):
        return self.env.env.check_success_forgiving()
    
    def render(self, camera_name, width, height):
        return self.env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)

    
    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        ft = obs['robot0_robot1_ft_denoised-state']

        # An environment is inviable if a force-torque reading exceeds maximum limit (or if env does not succeed)
        valid = True
        if np.max(ft[...,:3]) > 100.0 or np.max(ft[...,3:6]) > 6.0 or np.max(ft[...,6:9]) > 100.0 or np.max(ft[...,9:]) > 6.0:
            print(f'{PrintColorUtils.WARNING}[WARNING] Maximum force-torque threshold reached, skipping demo...{PrintColorUtils.ENDC}')
            valid = False
        
        return obs, valid

def clone_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    video_skip,
    camera_names=None,
    camera_height=84,
    camera_width=84
):
    assert isinstance(env, DataCollectionEnv)

    # get the initial state
    init_perturb = initial_state["init_perturb"]

    traj_len = states.shape[0]

    # maximum number of times to run through an environment
    max_envs = int((env.n_envs + 0.5 * env.n_envs)) + 1
    renders = []

    for i in range(max_envs):
        if env.canonical and i == 0:
            print(f'{PrintColorUtils.OKCYAN}[INFO] Recording canonical demo{PrintColorUtils.ENDC}')
        else:
            print(f'{PrintColorUtils.OKCYAN}[INFO] Recording clone {len(renders)}{PrintColorUtils.ENDC}')

        # load the initial state
        env.set_perturbation_values_from_array(init_perturb)

        # reactivate variations for non-canonical envs
        if not env.canonical or i > 0:
            env.unlock_variation_generators()
        
        env.reset()
        obs, _ = env.step(np.zeros(3))
        
        traj_renders = []

        # iteration variable @t is over "next obs" indices
        valid_demo = True
        frame = 0
        for t in tqdm(range(1, traj_len+1)):
            next_obs, valid = env.step(actions[t-1].copy())
            if frame % video_skip == 0:
                traj_renders.append(np.hstack([obs['closerenderview_image'], obs['left_wristview_image'][:,80:-80,:], obs['right_wristview_image'][:,80:-80,:]]))
            # traj_renders.append(obs['left_wristview_image'])
            # traj_renders.append(get_force_plot(obs['robot0_robot1_forcetorque-state']))
            valid_demo = valid_demo and valid

            if not valid:
                break
            
            r = env.get_reward()
            done = int(t == traj_len)

            # update for next iter
            obs = deepcopy(next_obs)

            frame += 1
        
        valid_demo = valid_demo and env.check_success()
        if valid_demo:
            renders.append(traj_renders)
        elif env.canonical and i == 0:
            print(f'{PrintColorUtils.WARNING}[WARNING] Canonical environment is inviable{PrintColorUtils.ENDC}, skipping demo...')
            return None
        
        if len(renders) == env.n_envs:
            # Stack observations for all trajectories
            render_all = []
            for step in range(len(renders[0])):
                render_all.append(np.vstack([render[step] for render in renders]))
            
            return render_all

    print(f'{PrintColorUtils.WARNING}[WARNING] Not enough environments were viable{PrintColorUtils.ENDC}, skipping demo...')
    return None

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

    print('[DEBUG] Creating video writer')
    video_writer = imageio.get_writer(os.path.join('video_visuals', args.video_name), fps=20)

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
        renders = clone_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
            video_skip=args.video_skip,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
        )

        # A failed trajectory clone will not be collected 
        if renders is None:
            print('[DEBUG] Restarting demo collection...')
            continue
        
        for img in renders:
            video_writer.append_data(img)

    f.close()
    video_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
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

    parser.add_argument(
        "--video_name",
        type=str,
        default='vis.mp4',
    )

    parser.add_argument(
        "--video_skip",
        type=int,
        default=1
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)