# This file contains commands that will help walk you through our full AugInsert pipeline (data collection, policy training, policy
# evaluation). It makes use of the provided set of human expert demonstrations, but you can easily substitute with your own collected
# dataset of trajectories.

# ======================================================================= #
# ============== STEP 0: Collect human expert trajectories ============== #
# ======================================================================= #

# First, record your demonstrations
python ctb_env/human_demo.py --record --name demo_example.hdf5

# Then, convert it into a Robomimic-compatible format
python robomimic/scripts/conversion/convert_robosuite.py --dataset demo_example.hdf5

# ===================================================================== #
# ============== STEP 1: Collect observations in dataset ============== #
# ===================================================================== #

# Dataset with only demonstrations in canonical environment
python robomimic/scripts/ctb_trajectory_cloning.py \
    --dataset ctb_data/datasets/demo_exp.hdf5 \
    --output_name train_wrist_canonical.hdf5 \
    --n_envs 1 \
    --camera_names left_wristview right_wristview \
    --ft_history_length 32 \
    --done_mode 2 \
    --exclude-next-obs \
    --canonical_env  \
    --p_seed 12345678

# Dataset with canonical environment demos + 12 augmentations involving a subset of
# Grasp Pose, Object Body Shape, and Peg/Hole Shape variations
python robomimic/scripts/ctb_trajectory_cloning.py \
    --dataset ctb_data/datasets/demo_exp.hdf5 \
    --output_name train_wrist_clone12.hdf5 \
    --n_envs 13 \
    --camera_names left_wristview right_wristview \
    --ft_history_length 32 \
    --done_mode 2 \
    --exclude-next-obs \
    --canonical_env \
    --obj_shape_vars key cross circle \
    --obj_body_shape_vars cube cylinder \
    --obj_vars xt zt zr \
    --p_seed 12345678

# ============================================================================== #
# ============== STEP 1.5 (optional): Visualize collected dataset ============== #
# ============================================================================== #

# Visualize the canonical for the train_wrist_canonical.hdf5 dataset collected in Step 1
python robomimic/scripts/ctb_visualize_dataset.py \
    --dataset ctb_data/datasets/train_wrist_canonical.hdf5 \
    --n_demos 10 \
    --n_stack 1 \
    --video_folder vis \
    --video_skip 5 \
    --render_image_names left_wristview_image right_wristview_image

# Visualize the canonical and 4 augmentations of the first 10 demonstrations for
# the train_wrist_clone12.hdf5 dataset collected in Step 1
python robomimic/scripts/ctb_visualize_dataset.py \
    --dataset ctb_data/datasets/train_wrist_clone12.hdf5 \
    --n_demos 10 \
    --n_stack 5 \
    --video_folder vis \
    --video_skip 5 \
    --render_image_names left_wristview_image right_wristview_image

# ================================================== #
# ============== STEP 2: Train policy ============== #
# ================================================== #

# Train the policy using the train_wrist_canonical.hdf5 dataset created in Step 1
# NOTE: If you collected additional augmentations in your dataset, you can include them
#       in the training process by increasing the 'num_traj_clones' parameter in the 
#       configs/ctb_base.json config file
python robomimic/scripts/train.py \
    --config configs/ctb_base.json \
    --dataset ctb_data/datasets/train_wrist_canonical.hdf5

# ======================================================================== #
# ============== STEP 3: Evaluate policy on task variations ============== #
# ======================================================================== #

# Evaluate trained policy on no-variations Canonical environment
python robomimic/scripts/ctb_rollout.py \
    --exp ctb_experiments \
    --policy auginsert_run \
    --n_rollouts 50 \
    --video_path_folder eval \
    --video_skip 5 \
    --camera_names frontview left_wristview \
    --set_canonical \
    --pose_vars trans \
    --seed 20262027 \
    --p_seed 20262027 

# Evaluate trained policy on a composition of all task variations
python robomimic/scripts/ctb_rollout.py \
    --exp ctb_experiments \
    --policy auginsert_run \
    --n_rollouts 50 \
    --video_path_folder eval \
    --video_skip 5 \
    --camera_names frontview left_wristview \
    --set_canonical \
    --obj_vars xt zt yr zr \
    --obj_shape_vars arrow line pentagon hexagon diamond u \
    --obj_body_shape_vars cube-thin cylinder-thin octagonal-thin \
    --visual_vars lighting texture arena-eval camera \
    --ft_noise_std 5.0 0.15 \
    --prop_noise_std 0.001 0.01 \
    --seed 20262027 \
    --p_seed 20262027
