import gymnasium as gym
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

    def record_demo_values(self, rgbs, left_f, right_f, left_t, right_t, show):
        self.times.append(self.time)

        self.forces_left.append(left_f)
        self.torques_left.append(left_t)

        self.forces_left = self.forces_left[-40:]
        self.torques_left = self.torques_left[-40:]

        self.forces_right.append(right_f)
        self.torques_right.append(right_t)

        self.forces_right = self.forces_right[-40:]
        self.torques_right = self.torques_right[-40:]

        if isinstance(rgbs, list):
            cur_rgb_frames = [rgb[:,:,[2,1,0]] for rgb in rgbs]
        else:
            cur_rgb_frames = [rgbs[:,:,[2,1,0]]]

        # l_cur_f_plot = self.get_current_force_plot_left()
        # l_cur_t_plot = self.get_current_torque_plot_left()

        r_cur_f_plot = self.get_current_force_plot_right()
        r_cur_t_plot = self.get_current_torque_plot_right()

        img = np.hstack(cur_rgb_frames + [r_cur_f_plot, r_cur_t_plot])

        if show:
            cv2.imshow('ft', img)
            cv2.waitKey(10)
        else:
            cv2.imwrite('imgs/swap_variation.png', img)

    def step(self, rgb, left_ft, right_ft, show=True):
        # Account for force-torque histories as inputs
        if left_ft.shape[0] > 0:
            left_ft = left_ft[-1]
        if right_ft.shape[0] > 0:
            right_ft = right_ft[-1]
            
        self.record_demo_values(
            rgb,
            left_ft[:3].copy(),
            right_ft[:3].copy(),
            left_ft[3:].copy(),
            right_ft[3:].copy(),
            show=show
        )
        self.time += 1

    def get_current_force_plot_left(self):
        fig = plt.figure(figsize=(3.2,2.4))
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
        fig = plt.figure(figsize=(3.2,2.4))
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
        fig = plt.figure(figsize=(3.2,2.4))
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
        fig = plt.figure(figsize=(3.2,2.4))
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