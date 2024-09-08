"""
Driver class for Keyboard controller.
"""

import numpy as np
from pynput.keyboard import Controller, Key, Listener

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

'''
    For human teleoperated translation-only demos controlling the left arm
'''
class DualArmKeyboard(Device):
    """
    A minimalistic driver class for a Keyboard.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print("===== LEFT ARM =====")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r-f", "move arm vertically")
        print_command("z-x", "rotate arm about x-axis")
        print_command("t-g", "rotate arm about y-axis")
        print_command("c-v", "rotate arm about z-axis")
        print("===== RIGHT ARM =====")
        print_command("u-h-j-k", "move arm horizontally in x-y plane")
        print_command("o-l", "move arm vertically")
        print_command("b-n", "rotate arm about x-axis")
        print_command("p-;", "rotate arm about y-axis")
        print_command("m-,", "rotate arm about z-axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # self.rotation_l = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # self.raw_drotation_l = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        # self.last_drotation_l = np.zeros(3)
        self.pos_l = np.zeros(3)  # (x, y, z)
        self.last_pos_l = np.zeros(3)
        self.grasp_l = False

        # self.rotation_r = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # self.raw_drotation_r = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        # self.last_drotation_r = np.zeros(3)
        self.pos_r = np.zeros(3)  # (x, y, z)
        self.last_pos_r = np.zeros(3)
        self.grasp_r = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """

        dpos_l = self.pos_l - self.last_pos_l
        self.last_pos_l = np.array(self.pos_l)
        # raw_drotation_l = (
        #     self.raw_drotation_l - self.last_drotation_l
        # )  # create local variable to return, then reset internal drotation
        # self.last_drotation_l = np.array(self.raw_drotation_l)

        dpos_r = self.pos_r - self.last_pos_r
        self.last_pos_r = np.array(self.pos_r)
        # raw_drotation_r = (
        #     self.raw_drotation_r - self.last_drotation_r
        # )  # create local variable to return, then reset internal drotation
        # self.last_drotation_r = np.array(self.raw_drotation_r)
        return dict(
            dpos_l=dpos_l,
            # rotation_l=self.rotation_l,
            # raw_drotation_l=raw_drotation_l,
            grasp_l=int(self.grasp_l),
            dpos_r=dpos_r,
            # rotation_r=self.rotation_r,
            # raw_drotation_r=raw_drotation_r,
            grasp_r=int(self.grasp_r),
            reset=self._reset_state,
        )

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position
            if key.char == "w":
                self.pos_l[0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "s":
                self.pos_l[0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "a":
                self.pos_l[1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "d":
                self.pos_l[1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "f":
                self.pos_l[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "r":
                self.pos_l[2] += self._pos_step * self.pos_sensitivity  # inc z

            elif key.char == "u":
                self.pos_r[0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "j":
                self.pos_r[0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "h":
                self.pos_r[1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "k":
                self.pos_r[1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "l":
                self.pos_r[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "o":
                self.pos_r[2] += self._pos_step * self.pos_sensitivity  # inc z

            # # controls for moving orientation
            # elif key.char == "z":
            #     drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
            #     self.rotation = self.rotation.dot(drot)  # rotates x
            #     self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            # elif key.char == "x":
            #     drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
            #     self.rotation = self.rotation.dot(drot)  # rotates x
            #     self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            # elif key.char == "t":
            #     drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
            #     self.rotation = self.rotation.dot(drot)  # rotates y
            #     self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            # elif key.char == "g":
            #     drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
            #     self.rotation = self.rotation.dot(drot)  # rotates y
            #     self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            # elif key.char == "c":
            #     drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
            #     self.rotation = self.rotation.dot(drot)  # rotates z
            #     self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            # elif key.char == "v":
            #     drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
            #     self.rotation = self.rotation.dot(drot)  # rotates z
            #     self.raw_drotation[2] -= 0.1 * self.rot_sensitivity

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for grasping
            if key == Key.space:
                self.grasp_l = not self.grasp_l  # toggle gripper
                self.grasp_r = not self.grasp_r

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()

        except AttributeError as e:
            pass

def dualinput2action(device, robots, env_configuration=None):
    """
    Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

    If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

    Args:
        device (Device): A device from which user inputs can be converted into actions. Can be either a Spacemouse or
            Keyboard device class

        robot (Robot): Which robot we're controlling

        active_arm (str): Only applicable for multi-armed setups (e.g.: multi-arm environments or bimanual robots).
            Allows inputs to be converted correctly if the control type (e.g.: IK) is dependent on arm choice.
            Choices are {right, left}

        env_configuration (str or None): Only applicable for multi-armed environments. Allows inputs to be converted
            correctly if the control type (e.g.: IK) is dependent on the environment setup. Options are:
            {bimanual, single-arm-parallel, single-arm-opposed}

    Returns:
        2-tuple:

            - (None or np.array): Action interpreted from @device including any gripper action(s). None if we get a
                reset signal from the device
            - (None or int): 1 if desired close, -1 if desired open gripper state. None if get a reset signal from the
                device

    """
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos_l, dpos_r, grasp_l, grasp_r, reset = (
        state["dpos_l"],
        state["dpos_r"],
        state["grasp_l"],
        state["grasp_r"],
        state["reset"],
    )

    # If we're resetting, immediately return None
    if reset:
        return None, None, None, None

    # Get controller reference
    controller_l = robots[0].controller
    gripper_dof_l = robots[0].gripper.dof

    controller_r = robots[1].controller
    gripper_dof_r = robots[1].gripper.dof

    # First process the raw drotation
    if controller_l.name == "OSC_POSE" and controller_r.name == "OSC_POSE":
        dpos_l = dpos_l * 75 if isinstance(device, DualArmKeyboard) else dpos_l * 125
        dpos_r = dpos_r * 75 if isinstance(device, DualArmKeyboard) else dpos_r * 125
    else:
        # No other controllers currently supported
        print("Error: Unsupported controller specified -- Robot must have either an IK or OSC-based controller!")

    # map 0 to -1 (open) and map 1 to 1 (closed)
    grasp_l = 1 if grasp_l else -1
    grasp_r = 1 if grasp_r else -1

    # Create action based on action space of individual robot
    action_l = np.concatenate([dpos_l, [grasp_l] * gripper_dof_l])
    action_r = np.concatenate([dpos_r, [grasp_r] * gripper_dof_r])

    # Return the action and grasp
    return action_l, action_r, grasp_l, grasp_r