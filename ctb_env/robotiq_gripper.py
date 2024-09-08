"""
6-DoF gripper with its open/close variant
"""
import numpy as np
import os

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class NewRobotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__("assets/gripper/2f85.xml", idn=idn)

    def format_action(self, action):
        return min(action, 0.0) * 255.0

    @property
    def init_qpos(self):
        return np.zeros((8,))

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_driver_collision",
                "left_coupler_collision",
                "left_spring_collision",
                "left_follower_collision",
                "left_pad1",
                "left_pad2"
            ],
            "right_finger": [
                "right_driver_collision",
                "right_coupler_collision",
                "right_spring_collision",
                "right_follower_collision",
                "right_pad1",
                "right_pad2"
            ],
            "left_fingerpad": ["left_pad1", "left_pad2"],
            "right_fingerpad": ["right_pad1", "right_pad2"],
        }


class NewRobotiq85Gripper(NewRobotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), 0, 1.0) * 255.0
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

class EVRobotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(os.path.dirname(__file__),"assets/gripper/robotiq_gripper_85_new.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        # return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])
        return np.array([-0.033, -0.108, -0.224, -0.033, -0.109, -0.224])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class EVRobotiq85Gripper(EVRobotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1