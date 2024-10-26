# AugInsert: Learning Robust Visual-Force Policies via Data Augmentation for Object Assembly Tasks

**Ryan Diaz<sup>1</sup>, Adam Imdieke<sup>1</sup>, Vivek Veeriah<sup>2</sup>, and Karthik Desingh<sup>1</sup>**

University of Minnesota<sup>1</sup>, Google DeepMind<sup>2</sup>

[**[Paper]**](https://arxiv.org/pdf/2410.14968) &ensp; [**[Video]**](https://www.youtube.com/watch?v=UTA7sefgs2o&t=1s) &ensp; [**[Project Page]**](https://rpm-lab-umn.github.io/auginsert/) 

This repository contains code for data collection, data augmentation, policy training, and policy evaluation on our object assembly task. Our work builds off of and 
borrows code from the [**Robosuite**](https://github.com/ARISE-Initiative/robosuite) and [**Robomimic**](https://github.com/ARISE-Initiative/robomimic) repositories, and we are grateful for their provided open-sourced resources.

## Table of Contents
[Setup](#setup)  
[Data Collection](#data-collection)  
[Policy Training](#policy-training)  
[Policy Evaluation](#policy-evaluation)  

---

### Setup
```
mamba create -n capthebottle python=3.10
mamba activate capthebottle
mamba install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python==4.9.0.80
pip install robosuite==1.4.1
pip install h5py==3.10.0
mamba install matplotlib
mamba install gymnasium==0.29.1
```

After cloning this repository into your environment, install the required dependencies in a Conda/Mamba environment:

```
cd auginsert

# Can also use conda in place of mamba
mamba env create -f environment.yml
```

Afterwards, activate the created Python environment:

```
# Can also use conda in place of mamba
mamba activate capthebottle
```

Finally, install the locally modified `robomimic` repository in the existing Mamba environment:

```
pip install -e .
```

---

### Data Collection

#### Human Expert Demonstrations

To collect human expert demonstrations in the no-variations *Canonical* environment,

```
python ctb_env/human_demo.py [--record] [--dual_arm] [--name NAME]
```

- `--record`: If provided, records successful trajectories in a .hdf5 file
- `--dual_arm`: If provided, enables dual-arm control (6-dimensional actions) (not used in the paper)
- `--name NAME`: If `--record` is provided, determines the name of the .hdf5 file created

The environment will automatically terminate and reset upon a successful demonstration. If you want to skip the current demonstration, press `q` and the simulation will reset without recording the current trajectory. To end the collection process, use `Ctrl+C` (note that the output hdf5 file will update for each new demonstration you record).

After recording the dataset, you must convert it into a format compatible with the Robomimic training framework (done in-place):

```
python robomimic/scripts/conversion/convert_robosuite.py --dataset PATH_TO_DATASET
```

If you would like to skip the demo recording process, we have also provided our own set of 57 human expert trajectories (50 for training, 7 for validation) off of which simulation datasets can be collected. This dataset can be found in `ctb_data/datasets/demo_exp.hdf5`. These trajectories were used in the experiments reported in our paper, so they are provided for reproducibility purposes.

#### Extracting (Augmented) Observations

Once a dataset of human expert trajectories is collected, we need to extract observations from these trajectories to create a dataset compatible with our training pipeline. This is also the step where online augmentation with subsets of our task variations can be applied. We do this using the `robomimic/scripts/ctb_trajectory_cloning.py` script. For example commands, refer to `Step 1` of the `pipeline_helper.sh` file.

The full list of arguments to apply task variations to the environment is provided below; note that these arguments are also applicable to the policy evaluation script (discussed in a later section):

- `--obj_vars`: Grasp Pose variations. Can be a subset of `[xt, zt, yr, zr]` (default `None`)
- `--obj_shape_vars`: Peg/Hole Shape variations. Can be a subset of `[arrow, line, pentagon, hexagon, diamond, u, key, cross, circle]` (default `key`)
- `--obj_body_shape_vars`: Object Body Shape variations. Can be a subset of `[cube, cylinder, octagonal, cube-thin, cylinder-thin, octagonal-thin]` (default `cube`)
- `--visual_vars`: Scene Appearance and Camera Pose variations. Can be a subset of `[lighting, texture, camera, arena-train, arena-eval]` (note that either `arena-train` or `arena-eval` can be provided, but not both) (default `None`)
- `--ft_noise_std`: Force-Torque Noise (part of Sensor Noise) variations. Given in the form `FORCE_STD TORQUE_STD`, which represent the standard deviations of Gaussian noise added to the corresponding input dimensions (default `0.0 0.0`)
- `--prop_noise_std`: Proprioceptive Noise (part of Sensor Noise) variations. Given in the form `POSITION_STD ROTATION_STD`, which represent the standard deviations of Gaussian noise added to the corresponding input dimensions (default `0.0 0.0`)

#### Visualizing Collected Datasets

After extracting observations, you can visualize the collected observations using the `robomimic/scripts/ctb_visualize_dataset.py` script. For an example command, refer to `Step 1.5` of the `pipeline_helper.sh` file.

---

### Policy Training

To train a policy using a collected dataset of extracted observations, use the `robomimic/scripts/train.py` script. We have provided the `configs/ctb_base.json` file containing training parameters that can be modified. For an example command, refer to `Step 2` of the `pipeline_helper.sh` file.

---

### Policy Evaluation

To evaluate a trained policy on a subset of task variations, use the `robomimic/scripts/ctb_rollout.py` script. For an example command, refer to `Step 3` of the `pipeline_helper.sh` file. Note that the task variation arguments detailed in the [Extracting Observations](#extracting-augmented-observations) section also applies here. 

You can also visualize attention maps during rollouts (as was done in a supplemental experiment shown on the website) by adding the `--visualize_attns` flag.
