import pandas as pd 
import wandb
import csv
import numpy as np
from ast import literal_eval
from collections import OrderedDict

TAG_TO_COLUMN = OrderedDict([
    ('canonical', 'Canonical'),
    ('grasp_eval', 'Eval Grasp Vars'),
    ('peg_hole_shape_eval', 'Eval Peg/Hole Shape'),
    ('obj_body_shape_eval', 'Eval Object Body Shape'),
    ('visual', 'Eval Visual Vars'),
    ('camera_angle', 'Camera Vars'),
    # ('ft_noise', 'FT Noise'),
    # ('prop_noise', 'Prop Noise'),
    ('sensor_noise', 'Sensor Noise'),
    # ('peg_hole_swap', 'Peg/Hole Swap'),
    # ('eval_vars', 'All Eval Vars'),
    ('no_swap', 'All Eval Vars (no swap)')
])

api = wandb.Api()
experiment = 'ablation_num_clones_wristviews_evals'

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

exp_results = {}
for summary, tags, name in zip(summary_list, tag_list, name_list):
    if tags[-1] in TAG_TO_COLUMN.keys():
        if name not in exp_results:
            exp_results[name] = {}
        success_rate_mean = summary['Success_Rate_Mean']
        success_rate_std = summary['Success_Rate_Std']
        exp_results[name][TAG_TO_COLUMN[tags[-1]]] = f'{success_rate_mean:.3f} +/- {success_rate_std:.3f}'

with open(f'results/{experiment}.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['Name'] + [v for v in TAG_TO_COLUMN.values()])
    for name, results in exp_results.items():
        w.writerow([name] + [results[v] for v in TAG_TO_COLUMN.values()])

                

