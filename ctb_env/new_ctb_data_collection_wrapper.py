import numpy as np
import json
import time
import os

from robosuite.wrappers import Wrapper, DataCollectionWrapper
from object_assembly_env import ObjectAssemblyInitializer

class ObjectAssemblyDataCollectionWrapper(DataCollectionWrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        if isinstance(env, Wrapper):
            assert isinstance(env.env, ObjectAssemblyInitializer), "This wrapper only works with a wrapped ObjectAssemblyInitializer environment!"
        else:
            assert isinstance(env, ObjectAssemblyInitializer), "This wrapper only works with the ObjectAssemblyInitializer environment!"

        super().__init__(env, directory, collect_freq, flush_freq)

        # Saving initial perturbations of episodes for deterministic playback
        self.init_perturb = None

        # Saving current proprioception state for recording delta actions
        self.current_prop = None

    def _start_new_episode(self):
        # super()._start_new_episode()
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())
        
        print("CTBDataCollectionWrapper: Recording init perturb states")
        self.init_perturb = self.env.get_perturbation_values_as_array()
        init_perturb_vis = {k: v.tolist() for k,v in self.env.get_perturbation_values().items()}
        print(json.dumps(init_perturb_vis, indent=4))

        self.current_prop = None

    # Overriding robosuite's _flush method
    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        # print('====FLUSHING====')

        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__

        init_perturb = None

        # Check that we're saving the init perturb state
        if self.init_perturb is not None:
            print("CTBDataCollectionWrapper: Saving init perturb states...")
            init_perturb = self.init_perturb

        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            init_perturb=init_perturb,
            env=env_name,
        )

        self.states = []
        self.action_infos = []
        self.successful = False
        self.init_perturb = None
    
    # Overriding robosuite's step method to log deltas as actions
    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = self.env.step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            rob_prop = ret[0]['robot0_eef_pos']
            if self.current_prop is None:
                self.current_prop = rob_prop

            # Save scaled (up) actions since they fit the [-1,1] range better
            delta_action = (rob_prop - self.current_prop) / self.env.trans_action_scale
            self.current_prop = rob_prop

            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            # Change recorded action to delta action
            info = {}
            info["actions"] = np.array(delta_action)
            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret

class DualArmObjectAssemblyDataCollectionWrapper(DataCollectionWrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        if isinstance(env, Wrapper):
            assert isinstance(env.env, ObjectAssemblyInitializer), "This wrapper only works with a wrapped ObjectAssemblyInitializer environment!"
        else:
            assert isinstance(env, ObjectAssemblyInitializer), "This wrapper only works with the ObjectAssemblyInitializer environment!"

        super().__init__(env, directory, collect_freq, flush_freq)

        # Saving initial perturbations of episodes for deterministic playback
        self.init_perturb = None

        # Saving current proprioception state for recording delta actions
        self.current_prop_left = None
        self.current_prop_right = None

    def _start_new_episode(self):
       # super()._start_new_episode()
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())
        
        print("CTBDataCollectionWrapper: Recording init perturb states")
        self.init_perturb = self.env.get_perturbation_values_as_array()
        init_perturb_vis = {k: v.tolist() for k,v in self.env.get_perturbation_values().items()}
        print(json.dumps(init_perturb_vis, indent=4))

        self.current_prop_left = None
        self.current_prop_right = None

    # Overriding robosuite's _flush method
    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        # print('====FLUSHING====')

        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__

        init_perturb = None

        # Check that we're saving the init perturb state
        if self.init_perturb is not None:
            print("CTBDataCollectionWrapper: Saving init perturb states...")
            init_perturb = self.init_perturb

        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            init_perturb=init_perturb,
            env=env_name,
        )

        self.states = []
        self.action_infos = []
        self.successful = False
        self.init_left_perturb = None
        self.init_right_perturb = None
    
    # Overriding robosuite's step method to log deltas as actions
    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = self.env.step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            rob_prop_l = ret[0]['robot0_eef_pos']
            rob_prop_r = ret[0]['robot1_eef_pos']
            if self.current_prop_left is None:
                self.current_prop_left = rob_prop_l
            if self.current_prop_right is None:
                self.current_prop_right = rob_prop_r

            # Save scaled (up) actions since they fit the [-1,1] range better
            delta_action_l = (rob_prop_l - self.current_prop_left) / self.env.trans_action_scale
            self.current_prop_left = rob_prop_l

            delta_action_r = (rob_prop_r - self.current_prop_right) / self.env.trans_action_scale
            self.current_prop_right = rob_prop_r

            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            # Change recorded action to delta action
            info = {}
            info["actions"] = np.concatenate((delta_action_l, delta_action_r))
            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret
