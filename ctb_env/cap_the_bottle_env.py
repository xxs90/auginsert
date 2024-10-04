import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import (
    find_elements,
    array_to_string,
    new_element
)
from robosuite.utils.observables import Observable, sensor, create_gaussian_noise_corrupter

try:
    from ctb_env.cap_bottle_obj import PegHoleObject
    from ctb_env.ctb_arena import CapTheBottleArena
    from ctb_env.ctb_domain_randomizer import CameraModder, LightingModder, TextureModder
except:
    from cap_bottle_obj import PegHoleObject
    from ctb_arena import CapTheBottleArena
    from ctb_domain_randomizer import CameraModder, LightingModder, TextureModder
from robosuite.models.objects import BoxObject

'''
    Base simulation environment for dual-arm peg-in-hole task
'''
class CapTheBottle(TwoArmEnv):
    def __init__(
        self,
        robots=["UR5e", "UR5e"],
        env_configuration="default",
        controller_configs=None,
        gripper_types=["Robotiq85Gripper", "Robotiq85Gripper"],
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=None,
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,         # Control frequency of task
        horizon=1000,            # Max steps per episode 
        ignore_done=False,
        hard_reset=True,
        camera_names="overhead",
        camera_heights=480,
        camera_widths=640,
        camera_depths=False,
        camera_segmentations=None,
        force_torque_hist_len=1, # History length of force-torque observations
        renderer="mujoco",
        renderer_config=None,
        **kwargs
    ):
        # For more readable terminal output
        np.set_printoptions(precision=3)

        # Save force-torque history length
        self.ft_hist_len = force_torque_hist_len

        # Set base poses for left and right arms
        self.base_left_pose = np.array([0, -0.0465, 0.625, -1.57, 0, 0, 1], dtype=np.float64) # 0, -0.0465, 0.625, -1.57, 0, 0, 1
        self.base_right_pose = np.array([0, 0.0465, 0.625, 1.57, 0, 0, 1], dtype=np.float64) # 0, 0.0465, 0.625, 1.57, 0, 0, 1

        # Set current poses for pseudo-delta OSC controller
        self.current_left_pose = self.base_left_pose[:6].copy()
        self.current_right_pose = self.base_right_pose[:6].copy()

        # Set base sim poses for left and right arms
        # self.base_robot0_qpos = [-1.6807,3.1554,2.1209,-2.1348,1.115e-1,8.4243e-04]
        # self.base_robot1_qpos = [1.6824,-0.0139,-2.1210,-1.0066,-0.1118,0]
        self.base_robot0_qpos = [-1.932,3.206,2.166,-2.230,3.631e-01,8.599e-04]
        self.base_robot1_qpos = [1.934,-6.427e-02,-2.166,-9.114e-01,-3.631e-01,-8.593e-04]

        # Set current sim poses for initialization
        self.init_robot0_qpos = self.base_robot0_qpos.copy()
        self.init_robot1_qpos = self.base_robot1_qpos.copy()

        # Standard dev values for Gaussian force-torque noise
        self.ft_noise_std = np.array([5.0, 0.05], dtype=np.float64)

        # Standard dev values for Gaussian proprioception noise
        self.prop_noise_std = np.array([0.001, 0.01], dtype=np.float64)

        # Peg/hole object vars for initialization
        # Set dummy vars for now
        self.peg = PegHoleObject(name="peg", type="peg", top_shape="diamond", body_shape="cube")
        self.hole = PegHoleObject(name="hole", type="hole", top_shape="diamond", body_shape="cube")

        self.peg_obj = self.peg.get_obj()
        self.hole_obj = self.hole.get_obj()

        self.base_obj_pose = np.array([0, 0, 0.015, 0.5, 0.5, 0.5, 0.5], dtype=np.float64) # z = 0.015

        # For square peg and hole
        # self.base_obj_pose = np.array([0, 0, -0.03, 0, 0, 0, 1])

        # Base pos/quat for objects
        self.peg_obj.set("pos", "0 0 0.01")
        self.peg_obj.set("quat", "0.5 0.5 0.5 0.5")
        self.hole_obj.set("pos", "0 0 0.01")
        self.hole_obj.set("quat", "0.5 0.5 0.5 0.5")

        # Scaling factors for actions
        self.trans_action_scale = 0.003
        self.rot_action_scale = 0.1

        # Arena texture (base is light-wood)
        self.arena_texture = 'light-wood'

        print("[CapTheBottle] Environment initialized")

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=[None,None],
            gripper_types=gripper_types,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
    
    # TODO: Change this depending on single-arm or dual-arm setup
    @property
    def action_spec(self):
        return np.zeros((3,)), np.ones((3,))
    
    # Override robosuite's edit_model_xml so that demo playbacks don't break
    def edit_model_xml(self, xml_str):
        return xml_str
    
    def reward(self, action=None):
        return 0
    
    def _load_model(self):
        # print("[CapTheBottle]: Loading model...")
        super()._load_model()

        # Setup robot arms
        for robot, offset in zip(self.robots, (-0.5, 0.5)):
            xpos = robot.robot_model.base_xpos_offset["empty"]
            xpos  = np.array(xpos) + np.array((0, offset, 0.5))
            robot.robot_model.set_base_xpos(xpos)
        
        self.robots[0].init_qpos = self.init_robot0_qpos
        self.robots[1].init_qpos = self.init_robot1_qpos

        self.robots[0].robot_model.set_base_ori(np.array((0,1.57,1.57)))
        self.robots[1].robot_model.set_base_ori(np.array((0,1.57,1.57)))

        mujoco_arena = CapTheBottleArena(texture=self.arena_texture)
        mujoco_arena.set_origin([0,0,0])

        # Default environment agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0666432116509934, 1.4903257668114777e-08, 2.0563394967349096],
            quat=[0.6530979871749878, 0.27104058861732483, 0.27104055881500244, 0.6530978679656982],
        )

        # Top-down camera
        mujoco_arena.set_camera(
            camera_name="overhead",
            pos=[0, 0, 1.325], # 1.4
            quat=[-0.71, 0, 0, 0.71]
        )

        # Front-view camera (for visualization purposes)
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[1, 0, 0.7],
            quat=[0.5, 0.5, 0.5, 0.5],
        )

        # Objects close-up (for visualization purposes)
        cam_rot = [0, -0.45, 0]
        mujoco_arena.set_camera(
            camera_name="closerenderview",
            pos=[0.35, 0, 0.8], # 1 0 0.7
            quat=T.quat_multiply([0.5, 0.5, 0.5, 0.5], T.mat2quat(T.euler2mat(cam_rot))),
        )

        # Attach wrist-view cameras to robots
        l_model, r_model = [self.robots[0].robot_model, self.robots[1].robot_model]

        camera_attribs = {}
        camera_attribs["pos"] = array_to_string([0.1,0,0])
        camera_attribs["quat"] = array_to_string(T.quat_multiply(np.array([0, 0.707108, 0.707108, 0]), T.mat2quat(T.euler2mat(np.array([0, -0.26, 0])))))
        camera_attribs["fovy"] = "75"
        
        # l_pref, r_pref = [self.robots[0].naming_prefix, self.robots[1].naming_prefix]
        l_eef = find_elements(root=l_model.worldbody, tags="body", attribs={"name": f"robot0_right_hand"})
        r_eef = find_elements(root=r_model.worldbody, tags="body", attribs={"name": f"robot1_right_hand"})
        l_eef.append(new_element(tag="camera", name="left_wristview", **camera_attribs))
        r_eef.append(new_element(tag="camera", name="right_wristview", **camera_attribs))

        # Attach peg and hole objects to gripper
        l_gripper, r_gripper = [self.robots[0].gripper, self.robots[1].gripper]
        l_pref, r_pref = l_gripper.naming_prefix, r_gripper.naming_prefix
        l_gripper_body = find_elements(root=l_gripper.worldbody, tags="body", attribs={"name": f"{l_pref}eef"})
        r_gripper_body = find_elements(root=r_gripper.worldbody, tags="body", attribs={"name": f"{r_pref}eef"})

        l_gripper_body.append(self.peg_obj)
        r_gripper_body.append(self.hole_obj)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=[self.peg, self.hole] <- Uncomment this for unattached objs
        )

        self.model.merge_assets(self.peg)
        self.model.merge_assets(self.hole)
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get robot prefix and define observables modality
        pf0 = self.robots[0].robot_model.naming_prefix
        pf1 = self.robots[1].robot_model.naming_prefix

        # Force-torque and proprioception information from both arms concatenated
        ft_all_denoised_modality = pf0 + pf1 + "ft_denoised"
        ft_all_modality = pf0 + pf1 + "forcetorque"
        prop_all_modality = pf0 + pf1 + "proprioception"

        # Force-torque data from both arms (with no noise applied), used during evaluation
        @sensor(modality=ft_all_denoised_modality)
        def ft_all_denoised(obs_cache):
            ft_curr = np.hstack([self.robots[0].ee_force, self.robots[0].ee_torque, self.robots[1].ee_force, self.robots[1].ee_torque])
            if "ft_all_denoised" in obs_cache.keys():
                ft_hist = obs_cache['ft_all_denoised']
                
                # Add current ft obs to history
                ft_hist = np.vstack((ft_hist, ft_curr))
                ft_hist = ft_hist[1:]
            else:
                # Pad history to match desired history length
                ft_hist = np.vstack((ft_curr, np.zeros(ft_curr.shape)))
                ft_hist = np.pad(ft_hist, pad_width=((self.ft_hist_len-ft_hist.shape[0]+1,0),), mode='edge')
                ft_hist = ft_hist[:-1, :12]
            return ft_hist

        # Force-torque data from both arms
        @sensor(modality=ft_all_modality)
        def ft_all(obs_cache):
            ft_curr = np.hstack([self.robots[0].ee_force, self.robots[0].ee_torque, self.robots[1].ee_force, self.robots[1].ee_torque])
            if "ft_all" in obs_cache.keys():
                ft_hist = obs_cache['ft_all']
                
                # Add current ft obs to history
                ft_hist = np.vstack((ft_hist, ft_curr))
                ft_hist = ft_hist[1:]
            else:
                # Pad history to match desired history length
                ft_hist = np.vstack((ft_curr, np.zeros(ft_curr.shape)))
                ft_hist = np.pad(ft_hist, pad_width=((self.ft_hist_len-ft_hist.shape[0]+1,0),), mode='edge')
                ft_hist = ft_hist[:-1, :12]
            return ft_hist

        # End-effector site position and rotation from both arms
        @sensor(modality=prop_all_modality)
        def prop_all(obs_cache):
            l_eef_pos = np.array(self.robots[0].sim.data.site_xpos[self.robots[0].eef_site_id])
            l_eef_quat = T.convert_quat(self.robots[0].sim.data.get_body_xquat(self.robots[0].robot_model.eef_name), to="xyzw")
            r_eef_pos = np.array(self.robots[1].sim.data.site_xpos[self.robots[1].eef_site_id])
            r_eef_quat = T.convert_quat(self.robots[1].sim.data.get_body_xquat(self.robots[1].robot_model.eef_name), to="xyzw")
            prop_curr = np.hstack([l_eef_pos, l_eef_quat, r_eef_pos, r_eef_quat])
            return prop_curr

        sensors = [ft_all_denoised, ft_all, prop_all]

        # Add in gaussian force-torque corrupter
        def ft_corrupter(inp):
            inp_c = np.array(inp)

            # Generate noise 
            force_noise = self.ft_noise_std[0] * np.random.randn(6)
            torque_noise = self.ft_noise_std[1] * np.random.randn(6)

            # Apply noise to measurements
            inp_c[-1,:3] += force_noise[:3]
            inp_c[-1,6:9] += force_noise[3:]
            inp_c[-1,3:6] += torque_noise[:3]
            inp_c[-1,9:] += torque_noise[3:]

            return inp_c
        
        # Add in gaussian proprioception corrupter
        def prop_corrupter(inp):
            inp_c = np.array(inp)

            # Generate noise
            pos_noise = self.prop_noise_std[0] * np.random.randn(6)
            rot_noise = self.prop_noise_std[1] * np.random.randn(6)

            # Apply noise to position measurement
            inp_c[:3] += pos_noise[:3]
            inp_c[7:10] += pos_noise[3:]

            # Apply noise to rotation measurement
            left_quat = inp_c[3:7]
            right_quat = inp_c[10:]
            inp_c[3:7] = T.quat_multiply(left_quat, T.mat2quat(T.euler2mat(rot_noise[:3])))
            inp_c[10:] = T.quat_multiply(right_quat, T.mat2quat(T.euler2mat(rot_noise[3:])))

            return inp_c
        
        corrupters = [None, ft_corrupter, prop_corrupter]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s, c in zip(names, sensors, corrupters):
            observables[name] = Observable(
                name=name,
                sensor=s,
                corrupter=c,
                sampling_rate=self.control_freq,
            )
        
        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
    
    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        diff = np.abs(peg_pos - hole_pos)
        success = (diff < np.array([0.00475, 0.0775, 0.00475])).all() # 0.07
        # print(success, 'diff', diff)
        return success

    def check_success_forgiving(self):
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        diff = np.abs(peg_pos - hole_pos)
        success = (diff < np.array([0.00475, 0.09, 0.00475])).all()
        # print(success, 'diff', diff)
        return success

    def step(self, action):
        action = action.copy()

        # 6-dim action -> Dual-arm translation-only
        if action.shape[0] == 6:
            left_action = action[:3] * self.trans_action_scale
            right_action = action[3:] * self.trans_action_scale

            # Update current pose for OSC controller
            self.current_left_pose[:3] += left_action
            self.current_right_pose[:3] += right_action
        
        # 3-dim action -> Single-arm translation-only
        elif action.shape[0] == 3:
            action *= self.trans_action_scale

            # Update current pose for OSC controller
            self.current_left_pose[:3] += action
        
        # 7-dim action -> Single-arm translation + rotation
        elif action.shape[0] == 7:
            action[:3] *= self.trans_action_scale
            action[3:6] *= self.rot_action_scale

            # Update current pose for OSC controller
            self.current_left_pose += action[:6]
        
        # 12-dim action -> Dual-arm translation + rotation
        else:
            left_action = action[:6]
            right_action = action[6:]

            left_action[:3] *= self.trans_action_scale
            right_action[:3] *= self.trans_action_scale
            left_action[3:] *= self.rot_action_scale
            right_action[3:] *= self.rot_action_scale

            self.current_left_pose += left_action
            self.current_right_pose += right_action
    
        # Converting euler angles to axis angle rotations for the OSC controller
        left_pose_rot = T.quat2axisangle(T.mat2quat(T.euler2mat(self.current_left_pose[3:6])))
        right_pose_rot = T.quat2axisangle(T.mat2quat(T.euler2mat(self.current_right_pose[3:6])))

        # Composing action with converted rotations
        left_action = np.concatenate((self.current_left_pose[:3], left_pose_rot, np.array([1])))
        right_action = np.concatenate((self.current_right_pose[:3], right_pose_rot, np.array([1])))
        action = np.hstack((left_action, right_action))

        return super().step(action)

'''
    Environment wrapper to generate random task initializations
'''
class CapTheBottleInitializer(CapTheBottle):
    ENV_PEG_HOLE_SHAPES = ['arrow', 'circle', 'cross', 'diamond', 'hexagon', 'key', 'line', 'pentagon', 'u']
    ENV_OBJ_BODY_SHAPES = ['cube', 'cylinder', 'octagonal', 'cube-thin', 'cylinder-thin', 'octagonal-thin']
    TRAIN_ARENA_TEXTURES = ['blue-wood', 'brass-ambra', 'ceramic', 'clay', 'dark-wood', 'light-wood'] # light-wood is default
    EVAL_ARENA_TEXTURES = ['cream-plaster',  'gray-felt', 'gray-plaster', 'gray-woodgrain', 'light-gray-floor-tile', 
                           'light-gray-plaster', 'metal', 'pink-plaster', 'red-wood', 'steel-brushed', 'steel-scratched', 
                           'wood-tiles', 'wood-varnished-panels', 'yellow-plaster'] 
    
    def __init__(
        self,
        obj_shape=None,                 # (str) If None, generates randomized peg/hole shape
        peg_body_shape=None,            # (str) If None, generates randomized peg body shape
        hole_body_shape=None,           # (str) If None, generates randomized hole body shape
        base_pose=None,                 # (list) If None, generates randomized base pose for both arms
        base_pose_perturbation=None,    # (list) If None, generates randomized left and right arm initial pos    
        peg_perturbation=None,          # (list) If None, generates randomized peg initial pos
        hole_perturbation=None,         # (list) If None, generates randomized hole initial pos
        peg_hole_swap=None,             # (bool) If None, randomly swaps the arms holding the peg and hole
        ft_noise_std=None,              # (list) If given, randomly applies Gaussian noise to force-torque measurements
        prop_noise_std=None,            # (list) If given, randomly applies Gaussian noise to proprioception measurements
        obj_shape_variations=None,      # (list) List of possible shapes if generating randomized peg/hole shapes
        obj_body_shape_variations=None, # (list) List of possible shapes if generating randomized object body shapes
        pose_variations=None,           # (list) Can be either "trans", "rot", or both if generating randomized poses
        obj_variations=None,            # (list) Can be a subset of ["xt", "zt", "yr", "zr"] if generating randomized obj poses
        visual_variations=None,         # (list) Can be a subset of ["camera", "lighting", "texture", "arena-train", "arena-eval"] for generating visual variations
        perturbation_seed=None,         # (int) Deterministic seed for randomized variations
        **kwargs                        # Arguments for CapTheBottle environment
    ):
        super().__init__(**kwargs)
        # Set initial perturbation variables (base left/right arm and ft noise stds are set in CapTheBottle)
        self.obj_shape = None
        self.peg_body_shape = None
        self.hole_body_shape = None
        self.left_perturb = None
        self.right_perturb = None
        self.peg_perturb = None
        self.hole_perturb = None
        self.peg_hole_swap = False

        # List of values to generate random values for
        self.randomized_values = []

        # Store given generation variations (or defaults if not given)
        self.obj_shape_variations = obj_shape_variations if obj_shape_variations is not None else CapTheBottleInitializer.ENV_PEG_HOLE_SHAPES
        self.obj_body_shape_variations = obj_body_shape_variations if obj_body_shape_variations is not None else CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES
        self.pose_variations = pose_variations if pose_variations is not None else ["trans"]
        self.obj_variations = obj_variations if obj_variations is not None else ["xt", "zt", "yr", "zr"]

        # Different (deterministic) seeds for generating poses and obj variations
        self.perturbation_seed = perturbation_seed
        if perturbation_seed is None:
            self.obj_perturbation_seed = perturbation_seed
        else:
            print(f"[CapTheBottleInitializer]: Using seed {perturbation_seed} for initializations")
            self.obj_perturbation_seed = perturbation_seed + 99

        # Initialize random generators
        self.rngs = {
            'base_pose_rng': np.random.default_rng(seed=self.perturbation_seed),            # Used for generating base insertion poses
            'pose_rng': np.random.default_rng(seed=self.perturbation_seed),                 # Used for generating initial arm poses
            'swap_rng': np.random.default_rng(seed=self.perturbation_seed),                 # Used for swapping peg/hole arms
            'noise_rng': np.random.default_rng(seed=self.perturbation_seed),                # Used for generating randomized ft noise stds
            'vis_rng': np.random.RandomState(seed=self.perturbation_seed),                  # Used for generating visual variations
            'obj_rng': np.random.default_rng(seed=self.obj_perturbation_seed),              # Used for generating grasp variations
            'obj_shape_rng': np.random.default_rng(seed=self.obj_perturbation_seed),        # Used for generating peg/hole shapes
            'obj_body_shape_rng': np.random.default_rng(seed=self.obj_perturbation_seed)    # Used for generating object body shapes
        }


        if obj_shape is not None:
            assert obj_shape in CapTheBottleInitializer.ENV_PEG_HOLE_SHAPES, f'Given peg/hole shape must be one of {CapTheBottleInitializer.ENV_PEG_HOLE_SHAPES}'
            self.obj_shape = obj_shape
        else:
            assert obj_shape_variations is not None, "You must set obj_shape_variations for randomized shapes!"
            self.randomized_values.append("obj_shape")
        
        if peg_body_shape is not None:
            assert peg_body_shape in CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES, f'Given peg shape must be one of {CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES}'
            self.peg_body_shape = peg_body_shape
        else:
            self.randomized_values.append("peg_body_shape")
        
        if hole_body_shape is not None:
            assert hole_body_shape in CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES, f'Given hole shape must be one of {CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES}'
            self.hole_body_shape = hole_body_shape
        else:
            self.randomized_values.append("hole_body_shape")
        
        if peg_hole_swap is not None:
            assert peg_hole_swap in [True, False]
            self.peg_hole_swap = peg_hole_swap
        else:
            self.randomized_values.append("peg_hole_swap")
        
        if base_pose is not None:
            # Should be 3 dim (pos)
            assert len(base_pose) == 3
            base_pose = np.array(base_pose, dtype=np.float64)
            base_left_pose = base_pose.copy()
            base_right_pose = base_pose.copy()
            base_left_pose[1] -= 0.0465
            base_right_pose[1] += 0.0465
            self.base_left_pose = np.concatenate((base_left_pose, np.array([-1.57,0,0,1], dtype=np.float64)))
            self.base_right_pose = np.concatenate((base_right_pose, np.array([1.57,0,0,1], dtype=np.float64)))
        else:
            self.randomized_values.append("base_pose")
        
        if base_pose_perturbation is not None:
            # Should be 2 x 6-dim (pos + euler angles)
            assert isinstance(base_pose_perturbation, list) and len(base_pose_perturbation) == 2
            left_pose_perturbation, right_pose_perturbation = base_pose_perturbation[0], base_pose_perturbation[1]
            assert len(left_pose_perturbation) == 6 and len(right_pose_perturbation) == 6
            self.left_perturb = np.array(left_pose_perturbation, dtype=np.float64)
            self.right_perturb = np.array(right_pose_perturbation, dtype=np.float64)
        else:
            # assert pose_variations is not None, "You must set pose_variations for randomized arm poses!"
            self.randomized_values.append("base_pose_perturb")
        
        if peg_perturbation is not None:
            # Should be 6-dim (pos + euler angles)
            assert isinstance(peg_perturbation, list) and len(peg_perturbation) == 6
            self.peg_perturb = np.array(peg_perturbation, dtype=np.float64)
        else:
            # assert obj_variations is not None, "You must set obj_variations for randomized object poses!"
            self.randomized_values.append("peg_perturb")
        
        if hole_perturbation is not None:
            # Should be 6-dim (pos + euler angles)
            assert isinstance(hole_perturbation, list) and len(hole_perturbation) == 6
            self.hole_perturb = np.array(hole_perturbation, dtype=np.float64)
        else:
            # assert obj_variations is not None, "You must set obj_variations for randomized object poses!"
            self.randomized_values.append("hole_perturb")
        
        if ft_noise_std is not None:
            assert len(ft_noise_std) == 2
            self.ft_noise_std = np.array(ft_noise_std, dtype=np.float64)
        else:
            self.ft_noise_std = np.array([0.0, 0.0], dtype=np.float64)
        
        if prop_noise_std is not None:
            assert len(prop_noise_std) == 2
            self.prop_noise_std = np.array(prop_noise_std, dtype=np.float64)
        else:
            self.prop_noise_std = np.array([0.0, 0.0], dtype=np.float64)
        
        # Set up visual variation generators
        self.domain_randomizers = {}
        if visual_variations is not None:
            if "arena-train" in visual_variations:
                assert "arena-eval" not in visual_variations, "You must choose one set of arena textures!"
                self.ARENA_TEXTURES = self.TRAIN_ARENA_TEXTURES
                self.randomized_values.append("arena_texture")
                visual_variations.remove("arena-train")
            elif "arena-eval" in visual_variations:
                self.ARENA_TEXTURES = self.EVAL_ARENA_TEXTURES
                self.randomized_values.append("arena_texture")
                visual_variations.remove("arena-eval")
                
            var_to_class = {
                'camera': CameraModder,
                'lighting': LightingModder,
                'texture': TextureModder
            }

            for var in visual_variations:
                self.domain_randomizers[var] = var_to_class[var](self.sim, random_state=self.rngs['vis_rng'])
            self.visual_vars_active = True
        else:
            self.visual_vars_active = False
        
        # Only reset environment if a step has been taken
        self.reset_counter = 0

        # Set up first episode
        self._setup_episode()

    # Upon initialization or reset, calculate and setup new init configuration for episode
    def _setup_episode(self):
        # Set new (randomized) initialization
        self._set_init_state()

        # Set base qpos and reset for new initialized state
        self.init_robot0_qpos = self.base_robot0_qpos.copy()
        self.init_robot1_qpos = self.base_robot1_qpos.copy()
        super().reset()

        # Setup arms in starting pose
        self._step_to_init_pose()

    # Determines the initial starting state for the episode
    def _set_init_state(self):
        # Get pose/obj/shape variations based on what is None
        if "obj_shape" in self.randomized_values:
            print("[CapTheBottleInitializer]: Object shape not given, generating random...")
            self.obj_shape = self._generate_randomized_obj_shape()
        else:
            print("[CapTheBottleInitializer]: Initializing with given object shape")
        
        if "peg_body_shape" in self.randomized_values:
            print("[CapTheBottleInitializer]: Peg body shape not given, generating random...")
            self.peg_body_shape = self._generate_randomized_obj_body_shape(peg=True)
        else:
            print("[CapTheBottleInitializer]: Initializing with given peg body shape")
        
        if "hole_body_shape" in self.randomized_values:
            print("[CapTheBottleInitializer]: Hole body shape not given, generating random...")
            self.hole_body_shape = self._generate_randomized_obj_body_shape(peg=False)
        else:
            print("[CapTheBottleInitializer]: Initializing with given peg body shape")
        
        if "peg_hole_swap" in self.randomized_values:
            print("[CapTheBottleInitializer]: Potentially swapping peg and hole...")
            self.peg_hole_swap = self._generate_randomized_peg_hole_swap()
        
        if "base_pose" in self.randomized_values:
            print("[CapTheBottleInitializer]: Base insertion pose not given, generating random...")
            self.base_left_pose, self.base_right_pose = self._generate_randomized_base_poses()
        else:
            print("[CapTheBottleInitializer]: Initializing with given base insertion pose")

        if "base_pose_perturb" in self.randomized_values:
            print("[CapTheBottleInitializer]: Left and right arm pose not given, generating random...")
            self.left_perturb, self.right_perturb = self._generate_randomized_poses()
        else:
            print("[CapTheBottleInitializer]: Initializing with given left and right arm pose")

        if "peg_perturb" in self.randomized_values:
            print("[CapTheBottleInitializer]: Peg pose not given, generating random...")
            self.peg_perturb = self._generate_randomized_obj_pose()

            # Adjust for thin peg body shape
            # TODO: Adjust for cylinder/octagonal?
            if 'thin' in self.peg_body_shape:
                self.peg_perturb[1] *= 0.6
        else:
            print("[CapTheBottleInitializer]: Initializing with given peg pose")
        
        if "hole_perturb" in self.randomized_values:
            print("[CapTheBottleInitializer]: Hole pose not given, generating random...")
            self.hole_perturb = self._generate_randomized_obj_pose()
        else:
            print("[CapTheBottleInitializer]: Initializing with given hole pose")
        
        if "arena_texture" in self.randomized_values:
            print("[CapTheBottleInitializer]: Randomizing arena texture...")
            self.arena_texture = self._generate_randomized_arena_texture()
        
        # Potentially swap peg and hole properties
        if self.peg_hole_swap:
            peg_type = "hole"
            hole_type = "peg"
            hole_body_shape, peg_body_shape = self.peg_body_shape, self.hole_body_shape
            hole_perturb, peg_perturb = self.peg_perturb, self.hole_perturb
        else:
            peg_type = "peg"
            hole_type = "hole"
            peg_body_shape, hole_body_shape = self.peg_body_shape, self.hole_body_shape
            peg_perturb, hole_perturb = self.peg_perturb, self.hole_perturb

        # Correct pose variation based on obj variations
        left_pose_correction, right_pose_correction = self._arm_perturb_from_obj(peg_perturb, hole_perturb)

        # Set pose variations
        self.current_left_pose = self.base_left_pose[:6] + self.left_perturb + left_pose_correction
        self.current_right_pose = self.base_right_pose[:6] + self.right_perturb + right_pose_correction

        # Set obj variations
        # Correct hinge pos and axis based on object variation (TODO: Correct for zr)
        peg_hinge = peg_perturb[:3].copy()
        hole_hinge = hole_perturb[:3].copy()
        peg_hinge[2] += self.base_obj_pose[2]   # Correcting base pose zt
        hole_hinge[2] += self.base_obj_pose[2]  # Correcting base pose zt
    
        self.peg = PegHoleObject(
            name=peg_type, 
            type=peg_type, 
            top_shape=self.obj_shape, 
            body_shape=peg_body_shape, 
            hinge_pos=" ".join([str(-a) for a in peg_hinge])
        )
        self.hole = PegHoleObject(
            name=hole_type, 
            type=hole_type, 
            top_shape=self.obj_shape, 
            body_shape=hole_body_shape, 
            hinge_pos=" ".join([str(-a) for a in hole_hinge])
        )

        self.peg_obj = self.peg.get_obj()
        self.hole_obj = self.hole.get_obj()

        new_peg_pos = self.base_obj_pose[:3] + peg_perturb[:3]
        new_peg_quat = T.quat_multiply(
            self.base_obj_pose[3:], 
            T.mat2quat(T.euler2mat(peg_perturb[3:]))
        )
        new_hole_pos = self.base_obj_pose[:3] + hole_perturb[:3]
        new_hole_quat = T.quat_multiply(
            self.base_obj_pose[3:], 
            T.mat2quat(T.euler2mat(hole_perturb[3:]))
        )

        self.peg_obj.set("pos", " ".join([str(p) for p in new_peg_pos]))
        self.peg_obj.set("quat", " ".join([str(p) for p in new_peg_quat]))
        self.hole_obj.set("pos", " ".join([str(p) for p in new_hole_pos]))
        self.hole_obj.set("quat", " ".join([str(p) for p in new_hole_quat]))

        print(
            f'''
            ====== [CapTheBottleInitializer]: Episode initialization ======
            obj_shape: {self.obj_shape}
            obj_body_shapes: {peg_body_shape} {hole_body_shape}
            peg_hole_swap: {self.peg_hole_swap}
            base_left_pose: {self.base_left_pose}                                  
            base_right_pose: {self.base_right_pose}
            left_perturb: {self.left_perturb}
            right_perturb: {self.right_perturb}
            peg_perturb: {peg_perturb}
            hole_perturb: {hole_perturb}
            ft_noise_stds: {self.ft_noise_std}
            prop_noise_stds: {self.prop_noise_std}
            ===============================================================
            '''
        )

    # Steps the arms to their initial perturbed state
    def _step_to_init_pose(self):
        print("[CapTheBottleInitializer] Setting initial robot pose...")

        # Pose before contact
        left_pose_pos = self.current_left_pose[:3].copy()
        left_pose_pos[1] -= 0.005
        right_pose_pos = self.current_right_pose[:3].copy()
        right_pose_pos[1] += 0.005

        # Converting euler angles to axis angle rotations for the OSC controller
        left_pose_rot = T.quat2axisangle(T.mat2quat(T.euler2mat(self.current_left_pose[3:6])))
        right_pose_rot = T.quat2axisangle(T.mat2quat(T.euler2mat(self.current_right_pose[3:6])))

        # Composing action with converted rotations
        left_action = np.concatenate((left_pose_pos, left_pose_rot, np.array([1])))
        right_action = np.concatenate((right_pose_pos, right_pose_rot, np.array([1])))

        # Moving to pre-contact pose
        action = np.hstack((left_action, right_action))
        for _ in range(40):
            super(CapTheBottle, self).step(action)

        # Composing action with converted rotations
        left_action = np.concatenate((self.current_left_pose[:3], left_pose_rot, np.array([1])))
        right_action = np.concatenate((self.current_right_pose[:3], right_pose_rot, np.array([1])))

        # Moving to contact pose
        action = np.hstack((left_action, right_action))
        for _ in range(15):
            super(CapTheBottle, self).step(action)
        
        # Set initial qpos for reset
        self.init_robot0_qpos = self.robots[0].sim.data.qpos[self.robots[0].joint_indexes]
        self.init_robot1_qpos = self.robots[1].sim.data.qpos[self.robots[1].joint_indexes]

        # These steps do not count for the reset counter
        self.reset_counter = 0

    # Returns a tuple of two 7-dim poses (xt,yt,zt,xr,yr,zr,gripper) with randomized translations
    def _generate_randomized_base_poses(self):
        # x, y, z axis translations
        random_base_pose_trans = self.rngs['base_pose_rng'].uniform(
            low=np.array([-0.025, -0.025, 0.6], dtype=np.float64),
            high=np.array([0.025, 0.025, 0.65], dtype=np.float64)
        )
        base_left_pose = np.concatenate((random_base_pose_trans, np.array([-1.57,0,0,1], dtype=np.float64)))
        base_right_pose = np.concatenate((random_base_pose_trans, np.array([1.57,0,0,1], dtype=np.float64)))
        base_left_pose[1] -= 0.0465
        base_right_pose[1] += 0.0465
        return base_left_pose, base_right_pose

    # Returns 2x6-dim randomized pose (xt,yt,zt,xr,yr,zr)
    def _generate_randomized_poses(self):
        # x, y, z axis translation offsets
        offsets = self.rngs['pose_rng'].uniform(
            low=np.array([0.015,0.0,0.015], dtype=np.float64),
            high=np.array([0.030,0.005,0.030], dtype=np.float64)
        )

        # Determine proportion of offset on each axis for both arms
        left_offsets_proportions = self.rngs['pose_rng'].uniform(low=0.0, high=1.0, size=(3,))
        left_pose_trans = offsets * left_offsets_proportions
        right_pose_trans = offsets - left_pose_trans

        # Randomly negate either left or right arm axes
        negates = self.rngs['pose_rng'].permuted(np.array([[1.0,-1.0],[1.0,-1.0],[1.0,-1.0]]), axis=1)
        left_pose_trans *= negates[:,0]
        right_pose_trans *= negates[:,1]

        # x, z axis rotations
        left_xz_rot = self.rngs['pose_rng'].uniform(low=-0.18, high=0.18, size=(2,))
        right_xz_rot = self.rngs['pose_rng'].uniform(low=-0.18, high=0.18, size=(2,))

        # y axis rotation
        left_y_rot = self.rngs['pose_rng'].choice(np.array([-1.5708, 0, 1.5708, 3.1415]))
        right_y_rot = self.rngs['pose_rng'].choice(np.array([-1.5708, 0, 1.5708, 3.1415]))

        left_pose_rot = np.array([left_xz_rot[0], left_y_rot, left_xz_rot[1]])
        right_pose_rot = np.array([right_xz_rot[0], right_y_rot, right_xz_rot[1]])

        # Filter out generations based on randomization settings
        if "trans" not in self.pose_variations:
            left_pose_trans[:] = 0.0
            right_pose_trans[:] = 0.0
        
        if "rot" not in self.pose_variations:
            left_pose_rot[:] = 0.0
            right_pose_rot[:] = 0.0
        
        left_random_pose = np.hstack([left_pose_trans, left_pose_rot], dtype=np.float64)
        right_random_pose = np.hstack([right_pose_trans, right_pose_rot], dtype=np.float64)

        return left_random_pose, right_random_pose

    # Returns 4-dim randomized pose (xt,zt,yr,zr)
    def _generate_randomized_obj_pose(self):
        random_xt_zt_yr = self.rngs['obj_rng'].uniform(
            low=np.array([-0.017, 0.0, -0.175]),
            high=np.array([0.017, 0.01425, 0.175])
        )

        # TODO: Leave out 3.1415 for now due to pose instability
        random_zr = self.rngs['obj_rng'].choice(np.array([-1.5708, 0, 1.5708], dtype=np.float64))

        # Filter out zr before correction
        if "zr" not in self.obj_variations:
            random_zr = 0.0

        random_obj_pose = np.append(random_xt_zt_yr, random_zr)

        if "xt" not in self.obj_variations:
            random_obj_pose[0] = 0.0
        
        if "zt" not in self.obj_variations:
            random_obj_pose[1] = 0.0
        
        if "yr" not in self.obj_variations:
            random_obj_pose[2] = 0.0
        
        random_obj_pose = np.array([0, random_obj_pose[0], random_obj_pose[1], random_obj_pose[3], 0, random_obj_pose[2]], dtype=np.float64)

        return random_obj_pose

    def _generate_randomized_obj_shape(self):
        shape = self.rngs['obj_shape_rng'].choice(self.obj_shape_variations)
        return shape

    def _generate_randomized_obj_body_shape(self, peg=False):
        shape = self.rngs['obj_body_shape_rng'].choice(self.obj_body_shape_variations)
        if not peg and 'thin' in shape:
            shape = shape.removesuffix('-thin')
        return shape
    
    def _generate_randomized_peg_hole_swap(self):
        return self.rngs['swap_rng'].choice([True, False])

    def _generate_randomized_arena_texture(self):
        return self.rngs['vis_rng'].choice(self.ARENA_TEXTURES)

    # Calculates the pose adjustment for both arms based on obj perturbations
    def _arm_perturb_from_obj(self, peg_variations, hole_variations):
        xt_peg, zt_peg, yr_peg, zr_peg = peg_variations[1], peg_variations[2], peg_variations[5], peg_variations[3]
        zt_peg += self.base_obj_pose[2]

        # Correct peg variation (compensating for yr rotation)
        zt_peg_arm = xt_peg * np.cos(-yr_peg) + zt_peg * np.sin(-yr_peg)
        yt_peg_arm = -(zt_peg * np.cos(-yr_peg) - xt_peg * np.sin(-yr_peg))
        xr_peg_arm = yr_peg
        zr_peg_arm = zr_peg

        peg_correction = np.array([0,yt_peg_arm,zt_peg_arm,xr_peg_arm,0,zr_peg_arm], dtype=np.float64)
        peg_correction = CapTheBottleInitializer._convert_perturb_zr(peg_correction, peg=True)

        xt_hole, zt_hole, yr_hole, zr_hole = hole_variations[1], hole_variations[2], hole_variations[5], hole_variations[3]
        zt_hole += self.base_obj_pose[2]

        # Correct hole variation (compensating for yr rotation)
        zt_hole_arm = -(xt_hole * np.cos(-yr_hole) + zt_hole * np.sin(-yr_hole))
        yt_hole_arm = zt_hole * np.cos(-yr_hole) + xt_hole * np.sin(yr_hole)
        xr_hole_arm = yr_hole
        zr_hole_arm = zr_hole

        hole_correction = np.array([0,yt_hole_arm,zt_hole_arm,xr_hole_arm,0,zr_hole_arm], dtype=np.float64)
        hole_correction = CapTheBottleInitializer._convert_perturb_zr(hole_correction, peg=False)

        return peg_correction, hole_correction

    @staticmethod
    def _convert_perturb_zr(p, peg=True):
        zr = ((p[-1] + 1.5708)) % (2*3.1415) - 1.5708
        zr_rots = np.array([-1.5708, 0, 1.5708, 3.1415], dtype=np.float64)
        zr_idx = np.abs(zr_rots-zr).argmin()

        if peg:
            if zr_idx == 0: # -90 
                p[0], p[2] = -p[2], p[0]
                p[3], p[4] = p[4], -p[3]
            elif zr_idx == 1: # 0
                p[0], p[2] = p[0], p[2]
                p[3], p[4] = p[3], p[4]
            elif zr_idx == 2: # 90
                p[0], p[2] = p[2], -p[0]
                p[3], p[4] = -p[4], p[3]
            elif zr_idx == 3: # 180
                p[0], p[2] = -p[0], -p[2]
                p[3], p[4] = -p[3], -p[4]
            else:
                raise Exception("Invalid zr rotation!")
        else:
            if zr_idx == 0: # -90 
                p[0], p[2] = p[2], -p[0]
                p[3], p[4] = p[4], -p[3]
            elif zr_idx == 1: # 0
                p[0], p[2] = p[0], p[2]
                p[3], p[4] = p[3], p[4]
            elif zr_idx == 2: # 90
                p[0], p[2] = -p[2], p[0]
                p[3], p[4] = -p[4], p[3]
            elif zr_idx == 3: # 180
                p[0], p[2] = -p[0], -p[2]
                p[3], p[4] = -p[3], -p[4]
            else:
                raise Exception("Invalid zr rotation!")

        return p
    
    # Set values of init state from numpy array representation
    def set_perturbation_values_from_array(self, init_perturb):
        base_pose = init_perturb[:3]
        left_perturb, right_perturb = init_perturb[3:9], init_perturb[9:15]
        peg_perturb, hole_perturb = init_perturb[15:21], init_perturb[21:27]
        obj_shape, peg_body_shape, hole_body_shape, swap = init_perturb[27], init_perturb[28], init_perturb[29], init_perturb[30]

        self.set_perturbation_values(
            base_pose=base_pose,
            base_pose_perturb=np.vstack((left_perturb,right_perturb)),
            peg_perturb=peg_perturb,
            hole_perturb=hole_perturb,
            obj_shape=obj_shape,
            peg_body_shape=peg_body_shape,
            hole_body_shape=hole_body_shape,
            swap=swap,
        )

    # Set values of init state; they are no longer randomized if given
    # Used for deterministic rollouts during data collection
    def set_perturbation_values(
        self,
        obj_shape: np.ndarray | str = None,
        peg_body_shape: np.ndarray | str = None,
        hole_body_shape: np.ndarray | str = None,
        base_pose: np.ndarray = None,
        base_pose_perturb: np.ndarray = None,
        peg_perturb: np.ndarray = None,
        hole_perturb: np.ndarray = None,
        swap: np.ndarray = None,
    ):
        if obj_shape is not None:
            if isinstance(obj_shape, str):
                assert obj_shape in CapTheBottleInitializer.ENV_PEG_HOLE_SHAPES
                self.obj_shape = obj_shape
            else:
                self.obj_shape = CapTheBottleInitializer.ENV_PEG_HOLE_SHAPES[int(obj_shape)]
            if "obj_shape" in self.randomized_values:
                self.randomized_values.remove("obj_shape")
        
        if peg_body_shape is not None:
            if isinstance(peg_body_shape, str):
                assert peg_body_shape in CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES
                self.peg_body_shape = peg_body_shape
            else:
                self.peg_body_shape = CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES[int(peg_body_shape)]
            if "peg_body_shape" in self.randomized_values:
                self.randomized_values.remove("peg_body_shape")
        
        if hole_body_shape is not None:
            if isinstance(hole_body_shape, str):
                assert hole_body_shape in CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES
                self.hole_body_shape = hole_body_shape
            else:
                self.hole_body_shape = CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES[int(hole_body_shape)]
            if "hole_body_shape" in self.randomized_values:
                self.randomized_values.remove("hole_body_shape")
        
        if swap is not None:
            self.peg_hole_swap = swap > 1e-8
            if "peg_hole_swap" in self.randomized_values:
                self.randomized_values.remove("peg_hole_swap")
        
        if base_pose is not None:
            assert base_pose.shape[0] == 3
            base_left_pose = base_pose.copy()
            base_right_pose = base_pose.copy()
            base_left_pose[1] -= 0.0465
            base_right_pose[1] += 0.0465
            self.base_left_pose = np.concatenate((base_left_pose, np.array([-1.57,0,0,1], dtype=np.float64)))
            self.base_right_pose = np.concatenate((base_right_pose, np.array([1.57,0,0,1], dtype=np.float64)))
            if "base_pose" in self.randomized_values:
                self.randomized_values.remove("base_pose")
        
        if base_pose_perturb is not None:
            assert base_pose_perturb.shape[0] == 2 and base_pose_perturb.shape[1] == 6
            self.left_perturb = base_pose_perturb[0]
            self.right_perturb = base_pose_perturb[1]
            if "base_pose_perturb" in self.randomized_values:
                self.randomized_values.remove("base_pose_perturb")
            
        if peg_perturb is not None:
            assert peg_perturb.shape[0] == 6
            self.peg_perturb = peg_perturb
            if "peg_perturb" in self.randomized_values:
                self.randomized_values.remove("peg_perturb")
        
        if hole_perturb is not None:
            assert hole_perturb.shape[0] == 6
            self.hole_perturb = hole_perturb
            if "hole_perturb" in self.randomized_values:
                self.randomized_values.remove("hole_perturb")
    
    def get_perturbation_values(self):
        base_pose = self.base_left_pose[:3].copy()
        base_pose[1] += 0.0465
        return {
            'base_pose': base_pose,
            'left_perturb': self.left_perturb,
            'right_perturb': self.right_perturb,
            'peg_perturb': self.peg_perturb,
            'hole_perturb': self.hole_perturb,
            'obj_shape': np.array([CapTheBottleInitializer.ENV_PEG_HOLE_SHAPES.index(self.obj_shape)], dtype=np.float64),
            'peg_body_shape': np.array([CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES.index(self.peg_body_shape)], dtype=np.float64),
            'hole_body_shape': np.array([CapTheBottleInitializer.ENV_OBJ_BODY_SHAPES.index(self.hole_body_shape)], dtype=np.float64),
            'peg_hole_swap': np.array([self.peg_hole_swap]),
        }

    def get_perturbation_values_as_array(self):
        return np.concatenate([v for v in self.get_perturbation_values().values()])

    def randomize_visuals(self):
        if len(self.domain_randomizers) <= 0:
            return
        print("[CapTheBottleInitializer] Generating visual variations")
        for dr in self.domain_randomizers.keys():
            self.domain_randomizers[dr].update_sim(self.sim)
            self.domain_randomizers[dr].randomize()

    def reset(self, seed=None):
        # Restart rngs with new seed if given
        if seed is not None:
            self.reset_perturbation_generator(seed=seed)

        # Only reset if steps have been taken in the current setup
        if self.reset_counter > 0:
            self._setup_episode()
        else:
            print("[CapTheBottleInitializer]: No steps taken, avoiding reset...")

        super().reset()
        if self.visual_vars_active:
            self.randomize_visuals()
    
    # Used for deterministic initializations during validation rollouts
    def reset_perturbation_generator(self, seed=None):
        if seed is not None:
            self.perturbation_seed = seed
            self.obj_perturbation_seed = seed + 99

        print("[CapTheBottleInitializer]: Resetting initial perturbation generator")
        for rng in self.rngs.keys():
            if 'obj' in rng:
                self.rngs[rng] = np.random.default_rng(seed=self.obj_perturbation_seed)
            else:
                self.rngs[rng] = np.random.default_rng(seed=self.perturbation_seed)
        
        # Also need to reset randomizers in visual variation generators
        self.rngs['vis_rng'] = np.random.RandomState(seed=self.perturbation_seed)
        for dr in self.domain_randomizers.keys():
            self.domain_randomizers[dr].random_state = self.rngs['vis_rng']

    # Updates reset_counter to avoid unnecessary resets
    def step(self, action):
        self.reset_counter += 1
        
        # NOTE: Emulating offline augmentation. Used in the "Expanded Visual+Noise" dataset (see website for details)
        # if self.visual_vars_active:
        #     self.randomize_visuals()
        
        # obs, x, y, z = super().step(action)

        # if self.visual_vars_active:
        #     obs['robot0_robot1_forcetorque-state'] *= np.random.uniform(0.1, 2.0)
        # return obs, x, y, z

        return super().step(action)

    