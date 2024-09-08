### Create an Environment

`mamba env create -f environment.yml`

### Collecting Human Demonstrations

`python human_demo.py`

To record demos, add the `--record` flag

For dual-arm demos, add the `--dual_arm` flag

### Trajectory Cloning

`python ctb_trajectory_cloning.py --folder PATH_TO_DEMO_FOLDER`

This script will play back your collected demonstrations on new sets of grasp and shape variations. Due to errors in grasp variation compensation, the instability of the UR5e arms in simulation, and different precision requirements across shapes, these new demonstrations may not always succeed. 
