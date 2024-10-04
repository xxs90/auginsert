# AugInsert: Learning Robust Visual-Force Policies via Data Augmentation for Object Assembly Tasks

**Ryan Diaz<sup>1</sup>, Adam Imdieke<sup>1</sup>, Vivek Veeriah <sup>2</sup>, and Karthik Desingh<sup>1</sup>**

University of Minnesota<sup>1</sup>, Google DeepMind<sup>2</sup>

[**[Project Page]**](https://rpm-lab-umn.github.io/auginsert/)

This repository contains code for data collection, data augmentation, policy training, and policy evaluation on our object assembly task. Our work builds off of and 
borrows code from the [**Robosuite**](https://github.com/ARISE-Initiative/robosuite) and [**Robomimic**](https://github.com/ARISE-Initiative/robomimic) repositories, and we are grateful for their provided open-sourced resources.

---

### Setup

After cloning this repository into your environment, install the required dependencies in a Conda/Mamba environment:

```
cd auginsert
mamba env install -f environment.yml
```

Afterwards, activate the created Python environment:

```
mamba activate capthebottle
```

Finally, install the locally modified `robomimic` repository in the existing Mamba environment:

```
pip install -e .
```

---

### Data Collection

#### Human Expert Demonstrations

To collect human expert demonstrations,

```
python ctb_env/human_demo.py [--record] [--dual_arm] [--name NAME]
```

We have also provided a set of 57 human expert trajectories (50 for training, 7 for validation) off of which simulation datasets can be collected. This dataset can be found in `ctb_data/datasets/demo_exp.hdf5`. These trajectories were used in the experiments reported in our paper, so they are provided for reproducibility purposes.

#### Collecting Augmented Demonstrations

TODO: Trajectory cloning / online augmentation

#### Visualizing Collected Datasets

TODO: Dataset visualization

---

### Policy Training

TODO: 

---

### Policy Evaluation

TODO: Evaluate script
