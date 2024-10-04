import pandas as pd 
import wandb
import csv
import numpy as np
from ast import literal_eval
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42

TAG_TO_COLUMN = OrderedDict([
    # ('canonical', 'Canonical'),
    # ('grasp_eval', 'Grasp\nPose'),
    # ('peg_hole_shape_eval', 'Peg/Hole\nShape'),
    # ('obj_body_shape_eval', 'Object Body\nShape'),
    ('visual', 'Scene\nAppearance'),
    ('camera_angle', 'Camera\nPose'),
    # ('ft_noise', 'Force-Torque\nNoise'), # excluded
    # ('prop_noise', 'Proprioception\nNoise'), # excluded
    ('sensor_noise', 'Sensor\nNoise'),
    # ('peg_hole_swap', 'Peg/Hole\nSwap'), # excluded
    # ('eval_vars', 'All\nVariations'), # excluded
    ('no_swap', 'All Variations')
])

BREAK_TO_NONE = OrderedDict([
    ('Canonical', 'Canonical'),
    ('Grasp\nPose', 'Grasp Pose'),
    ('Peg/Hole\nShape', 'Peg/Hole Shape'),
    ('Object Body\nShape', 'Object Body Shape'),
    ('Scene\nAppearance', 'Scene Appearance'),
    ('Camera\nPose', 'Camera Pose'),
    ('Force-Torque\nNoise', 'Force-Torque Noise'),
    ('Prop.\nNoise', 'Prop. Noise'),
    ('Sensor\nNoise', 'Sensor Noise'),
    ('Peg/Hole\nSwap', 'Peg/Hole Swap'),
    ('All Variations', 'All Variations')
])

def get_data(experiment):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"diaz0329/{experiment}")

    summary_list, tag_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .name is the human-readable name of the run.
        name_list.append(run.name)

        tag_list.append(run.tags)

    exp_results_mean = OrderedDict()
    exp_results_std = OrderedDict()

    for summary, tags, name in zip(summary_list, tag_list, name_list):
        if tags[-1] in TAG_TO_COLUMN:
            if name not in exp_results_mean:
                exp_results_mean[name] = {}
                exp_results_std[name] = {}
            try:
                success_rate_mean = summary['Success_Rate_Mean']
                success_rate_std = summary['Success_Rate_Std']
                exp_results_mean[name][TAG_TO_COLUMN[tags[-1]]] = success_rate_mean
                exp_results_std[name][TAG_TO_COLUMN[tags[-1]]] = success_rate_std
            except:
                raise Exception('[ERROR] Success rates not found; you may have to wait for evaluations to finish running!')
    
    labels = list(exp_results_mean.keys())
    categories = list(TAG_TO_COLUMN.values())
    results_mean = [[exp_results_mean[l][c] for c in categories] for l in labels]
    results_std = [[exp_results_std[l][c] for c in categories] for l in labels]
    
    return labels, categories, results_mean, results_std

def get_radar_plot(exp_name, labels, categories, means, stds):
    # Number of variables we're plotting.
    num_vars = len(categories)

    # Colors of plots
    colors = ['#1aaf6c', '#429bf4', '#d42cea', 'red', 'orange', 'yellow', 'purple', 'mediumvioletred'][:len(labels)]

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    # Helper function to plot each car on the radar chart.
    def add_to_radar(data, name, color):
        data += data[:1]
        ax.plot(angles, data, color=color, linewidth=2.0, label=name)
        ax.fill(angles, data, color=color, alpha=0.1)
    
    NAME_TO_LABEL = {
        'prop_on_ft_on_rgb-wrist_on': 'Full Model',
        'prop_off_ft_on_rgb-wrist_on': 'No Prop.',
        'prop_on_ft_off_rgb-wrist_on': 'No Touch',
        'prop_on_ft_on_rgb-wrist_off': 'No Vision',
        'prop_off_ft_off_rgb-wrist_on': 'Vision Only'
    }

    names = [l.split('tclone6_')[-1] for l in labels]
    label_idxs = [names.index(name) for name in NAME_TO_LABEL.keys()]

    # Add each model to the chart.
    for idx in label_idxs:
        add_to_radar(means[idx], NAME_TO_LABEL[names[idx]], colors[idx])

    # # Add each model to the chart.
    # for data, name, color in zip(means, labels, colors):
    #     name = name.split('tclone6_')[-1]
    #     add_to_radar(data, NAME_TO_LABEL[name], color)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    categories += categories[:1]
    ax.set_thetagrids(np.degrees(angles), categories)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure radar goes from 0 to 100.
    ax.set_ylim(0.0, 1.0)

    # You can also set gridlines manually like this:
    # ax.set_rgrids([20, 40, 60, 80, 100])

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    ax.set_rlabel_position(180 / num_vars)

    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=12)
    # Change styling of factor labels
    ax.tick_params(axis='x', labelsize=16, pad=15)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    # ax.spines['polar'].set_color('#222222')
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FAFAFA')

    # # Add title.
    # ax.set_title('Test Plot', y=1.08)

    # Add a legend as well.
    ax.legend(loc='upper right', ncol=1, prop={'size': 11}, bbox_to_anchor=(1.4, 1.15))

    plt.savefig(f'results/{exp_name}.pdf', bbox_inches='tight')

def get_bar_plot(exp_name):
    # NAME_TO_LABEL = {
    #     'canonical': 'No Augmentations',
    #     # 'train': 'Base (Grasp+P/H/Body Shape)',
    #     # 'train_vis': 'Base+Visual',
    #     # 'train_vis_noise': 'Base+Visual+Sensor Noise'
    # }
    # NAME_TO_LABEL = {
    #     'prop_on_ft_on_rgb-wrist_on': 'Full Model',
    #     'prop_off_ft_on_rgb-wrist_on': 'No Prop.',
    #     'prop_on_ft_off_rgb-wrist_on': 'No Touch',
    #     'prop_on_ft_on_rgb-wrist_off': 'No Vision',
    #     'prop_off_ft_off_rgb-wrist_on': 'Vision Only'
    # }
    NAME_TO_LABEL = OrderedDict([
        ('prop_on_ft_on_rgb-wrist_on', 'Full Model'),
        ('prop_off_ft_on_rgb-wrist_on', 'No Prop.'),
        ('prop_on_ft_off_rgb-wrist_on', 'No Touch'),
        ('prop_on_ft_on_rgb-wrist_off', 'No Vision'),
        ('prop_off_ft_off_rgb-wrist_on', 'Vision Only')
    ])
    categories = [
        'Canonical',
        # 'Grasp Pose\n(XT)',
        'Grasp Pose\n(XT+ZR)',
        'Object Body\nShape',
        # 'Object Color',
        # 'Object Color\n+ Lighting'
    ]

    fig, ax = plt.subplots(figsize=(10, 4)) # 10, 4
    ax.set_ylim(0.0, 1.01)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.set_ylabel('Success Rate', fontsize=15)

    x = np.arange(len(categories))
    # num_bars = len(means)
    width = 0.15 # width of bars # 0.5
    multiplier = 0

    num_bars = len(NAME_TO_LABEL.keys())

    # Order labels based on desired models
    # names = [l.split('dataset_')[-1] for l in labels]
    # names = [l.split('tclone6_')[-1] for l in labels]
    # label_idxs = [names.index(name) for name in NAME_TO_LABEL.keys()]
    # means = np.array([[0.9, 0.2, 0.15, 0.8, 0.85, 0.8]])
    means = np.array([
        [0.9, 0.15, 0.8],
        [1.0, 0.4, 0.95],
        [0.2, 0.0, 0.0],
        [0.7, 0.05, 0.8],
        [0.05, 0.1, 0.1]
    ])

    # rects = ax.bar(x, means, width=width, color='lightcoral')

    for idx in range(len(means)):
        offset = width * multiplier
        rects = ax.bar(x+offset, means[idx], width=width, label=list(NAME_TO_LABEL.values())[idx]) #, color='lightcoral')
        ax.bar_label(rects, label_type='center', fmt='%.2f')
        multiplier += 1

    # for data_mean, data_std, name, color in zip(means, stds, labels, colors):
    #     name = name.split('dataset_')[-1]
    #     offset = width * multiplier
    #     rects = ax.bar(x+offset, data_mean, width=width, yerr=data_std, ecolor='black', capsize=2.5, label=NAME_TO_LABEL[name])#, color=color)
    #     # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    
    if num_bars % 2 == 0:
        ax.set_xticks(x-(0.5*width)+((num_bars // 2)*width), categories)
    else:
        ax.set_xticks(x+(num_bars // 2) * width, categories)
    ax.tick_params('x', length=0, labelsize=11)
    # ax.legend(loc='lower center', prop={'size': 10.5}, ncols=len(labels), bbox_to_anchor=(0.5, -0.25))
    ax.legend(loc='lower center', prop={'size': 10}, ncols=5, bbox_to_anchor=(0.5, -0.25))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Dataset with Human-Only Demonstrations (No Augmentation)', fontsize=15, y=1)
    plt.savefig(f'results/{exp_name}.pdf', bbox_inches='tight')

def get_line_plot(exp_name, labels, categories, means, stds):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylim(0.0, 1.01)
    ax.set_ylabel('Success Rate', fontsize=15)
    ax.set_xlabel('Augmentations per Human Demo', fontsize=15) # trajectory clones
    # ax.set_xlabel('Number of Latents', fontsize=15) # num latents
    # ax.set_xlabel('Token Dropout Rate', fontsize=15) # token dropout

    # Colors of plots
    colors = ['#1aaf6c', '#429bf4', '#d42cea', 'red', 'orange', 'green', 'purple', 'mediumvioletred'][:len(categories)]

    # Need to transpose data: labels as x axis, categories as lines
    # Sort labels based on numerical quantity of interest
    values = [int(l.split('_')[-1].split('clones')[0]) for l in labels] # Trajectory clones
    # values = [int(l.split('latents')[-1]) for l in labels] # Num latents
    # values = [int(l.split('drop')[-1]) for l in labels] # Token dropout
    idxs = np.argsort(np.array(values))
    x = np.arange(len(values))

    for i, name in enumerate(categories):
        if 'All Variations' in name:
            ax.plot(
                x,
                [means[idx][i] for idx in idxs], 
                label=BREAK_TO_NONE[name],
                color=colors[i],
                marker='D',
                linestyle='dashdot',
                linewidth=2,
                markersize=6
            )
        else:
            ax.plot(
                x,
                [means[idx][i] for idx in idxs], 
                label=BREAK_TO_NONE[name],
                color=colors[i],
                marker='o',
                linestyle='dotted',
                linewidth=2,
                markersize=6
            )
        ax.errorbar(
            x,
            [means[idx][i] for idx in idxs],
            yerr=[stds[idx][i] for idx in idxs],
            color=colors[i],
            linestyle='',
            marker='',
            linewidth=2
        )
    
    ax.set_xticks(x, sorted(values))
    ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', labelsize=12)

    # Add a legend as well.
    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_marker("")
        handle.set_linestyle("solid")

    ax.legend(loc='lower right', prop={'size': 10.5}, handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)}, ncols=1)#, bbox_to_anchor=(1.45, 0.4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_title('Training Set Variations', fontsize=15, y=1.05)
    ax.set_title('Evaluation Set Variations', fontsize=15, y=1.05)

    plt.savefig(f'results/{exp_name}.pdf', bbox_inches='tight')


def get_plot(data, plot_type, exp_name):
    if plot_type == 'radar':
        get_radar_plot(exp_name, *data)
    elif plot_type == 'line':
        get_line_plot(exp_name, *data)
    elif plot_type == 'bar':
        get_bar_plot(exp_name, *data)

def main():
    experiment = 'ablation_real_world_modinput'
    get_bar_plot(experiment)
    print('Saved!')
    print(experiment)

main()
