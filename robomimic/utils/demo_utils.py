import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

# Records and saves relevant values when running/recording scripted demonstrations
class DemoRecorder:
    def __init__(self):
        self.time = 0
        self.times = []

        self.forces_left = []
        self.torques_left = []

        self.forces_right = []
        self.torques_right = []

        self.prop_left = []
        self.prop_right = []

        self.rgb_frames = {
            'overhead': [],
            'wrist_left': [],
            'wrist_right': []
        }

        self.video_frames = []

    def reset(self):
        self.time = 0
        self.times = []

        self.forces_left = []
        self.torques_left = []

        self.forces_right = []
        self.torques_right = []

        self.prop_left = []
        self.prop_right = []

        self.rgb_frames = {
            'overhead': [],
            'wrist_left': [],
            'wrist_right': []
        }

        self.video_frames = []

    def _record_demo_values(self, rgbs, left_f, right_f, left_t, right_t, action):
        self.times.append(self.time)

        self.forces_left.append(left_f)
        self.torques_left.append(left_t)

        self.forces_left = self.forces_left[-40:]
        self.torques_left = self.torques_left[-40:]

        self.forces_right.append(right_f)
        self.torques_right.append(right_t)

        self.forces_right = self.forces_right[-40:]
        self.torques_right = self.torques_right[-40:]

        cur_rgb_frames = rgbs

        l_cur_f_plot = self.get_current_force_plot_left()
        l_cur_t_plot = self.get_current_torque_plot_left()

        # r_cur_f_plot = self.get_current_force_plot_right()
        # r_cur_t_plot = self.get_current_torque_plot_right()

        plots = [l_cur_f_plot, l_cur_t_plot]

        if action is not None:
            act_plot = self.get_current_act_plot(action)
            plots.append(act_plot)

        img = np.hstack(cur_rgb_frames + plots)

        return img

    def step(self, rgb, left_ft, right_ft, action=None):
        # Account for force-torque histories as inputs
        if len(left_ft.shape) > 1 and left_ft.shape[0] > 0:
            left_ft = left_ft[-1]
        if len(right_ft.shape) > 1 and right_ft.shape[0] > 0:
            right_ft = right_ft[-1]
        
        if action is not None:
            action = action.copy()
            
        img = self._record_demo_values(
            rgb,
            left_ft[:3].copy(),
            right_ft[:3].copy(),
            left_ft[3:].copy(),
            right_ft[3:].copy(),
            action,
        )
        self.time += 1

        return img

    def get_current_force_plot_left(self):
        fig = plt.figure(figsize=(6.4,4.8))
        plt.title("Left Arm Forces")
        plt.ylim(-200, 200)
        plt.plot(self.times[-40:], [x[0] for x in self.forces_left][-40:], linestyle="-", marker=".", markersize=1, color="r", label="force-x")
        plt.plot(self.times[-40:], [x[1] for x in self.forces_left][-40:], linestyle="-", marker=".", markersize=1, color="g", label="force-y")
        plt.plot(self.times[-40:], [x[2] for x in self.forces_left][-40:], linestyle="-", marker=".", markersize=1, color="b", label="force-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data

    def get_current_torque_plot_left(self):
        fig = plt.figure(figsize=(6.4,4.8))
        plt.title("Left Arm Torques")
        plt.ylim(-3, 3)
        plt.plot(self.times[-40:], [x[0] for x in self.torques_left][-40:], linestyle="-", marker=".", markersize=1, color="r", label="torque-x")
        plt.plot(self.times[-40:], [x[1] for x in self.torques_left][-40:], linestyle="-", marker=".", markersize=1, color="g", label="torque-y")
        plt.plot(self.times[-40:], [x[2] for x in self.torques_left][-40:], linestyle="-", marker=".", markersize=1, color="b", label="torque-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data
    
    def get_current_force_plot_right(self):
        fig = plt.figure(figsize=(6.4,4.8))
        plt.title("Right Arm Forces")
        plt.ylim(-200, 200)
        plt.plot(self.times[-40:], [x[0] for x in self.forces_right][-40:], linestyle="-", marker=".", markersize=1, color="r", label="force-x")
        plt.plot(self.times[-40:], [x[1] for x in self.forces_right][-40:], linestyle="-", marker=".", markersize=1, color="g", label="force-y")
        plt.plot(self.times[-40:], [x[2] for x in self.forces_right][-40:], linestyle="-", marker=".", markersize=1, color="b", label="force-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data

    def get_current_torque_plot_right(self):
        fig = plt.figure(figsize=(6.4,4.8))
        plt.title("Right Arm Torques")
        plt.ylim(-3, 3)
        plt.plot(self.times[-40:], [x[0] for x in self.torques_right][-40:], linestyle="-", marker=".", markersize=1, color="r", label="torque-x")
        plt.plot(self.times[-40:], [x[1] for x in self.torques_right][-40:], linestyle="-", marker=".", markersize=1, color="g", label="torque-y")
        plt.plot(self.times[-40:], [x[2] for x in self.torques_right][-40:], linestyle="-", marker=".", markersize=1, color="b", label="torque-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data
    
    def get_current_act_plot(self, act):
        fig, ax_act = plt.subplots(figsize=(6.4, 4.8))
        ax_act.set_ylim(-1, 1)
        ax_act.set_xticks(np.arange(3))
        ax_act.set_xticklabels(["x", "y", "z"])
        ax_act.bar(np.arange(3), act)

        ax_act.set_ylabel('output')
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data

class VTTDemoRecorder:
    def __init__(self):
        self.rgb_attn = [0.0] * 32
        self.ft_attn = [0.0] * 32

    @staticmethod
    def get_force_attn_plot(forces, attns, hist_len=32):
        fig, ax_f = plt.subplots(figsize=(6.4, 4.8))
        ax_f.set_ylim(-200, 200)
        ax_attn = ax_f.twinx()
        ax_attn.set_ylim(0, 1)
        ax_f.plot(np.arange(1,33), [x[0] for x in forces], linestyle='-', marker=".", markersize=1, color="r", label="force-x")
        ax_f.plot(np.arange(1,33), [x[1] for x in forces], linestyle='-', marker=".", markersize=1, color="g", label="force-y")
        ax_f.plot(np.arange(1,33), [x[2] for x in forces], linestyle='-', marker=".", markersize=1, color="b", label="force-z")
        ax_f.legend(loc="upper right")

        ax_attn.bar(np.arange(33-hist_len,33), attns)

        ax_f.set_ylabel('forces')
        ax_attn.set_ylabel('proportion of attention')

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data
    
    def get_attn_proportion_plot(self, proportions):
        self.rgb_attn.append(proportions[0])
        self.ft_attn.append(proportions[1])

        fig = plt.figure(figsize=(6.4,4.8))
        plt.title("Proportion of Modality-Specific Attentions")
        plt.ylim(0, 1)
        plt.plot(np.arange(1,33), self.rgb_attn[-32:], linestyle="-", marker=".", markersize=1, color="c", label="img attn")
        plt.plot(np.arange(1,33), self.ft_attn[-32:], linestyle="-", marker=".", markersize=1, color="m", label="ft attn")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.reshape((480,640,3))

        plt.close()

        return data



