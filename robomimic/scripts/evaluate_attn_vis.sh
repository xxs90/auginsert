# Canonical
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --seed 20262027 --p_seed 20262027 --visualize_attns

# Eval grasp vars
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --obj_vars xt zt yr zr --seed 20262027 --p_seed 20262027 --visualize_attns

# Eval peg/hole shape
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --obj_shape_vars arrow line pentagon hexagon diamond u --seed 20262027 --p_seed 20262027 --visualize_attns

# Eval body shape
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --obj_body_shape_vars cube-thin cylinder-thin octagonal-thin --seed 20262027 --p_seed 20262027 --visualize_attns

# Visual vars
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --visual_vars lighting texture arena-eval --seed 20262027 --p_seed 20262027 --visualize_attns

# Camera vars
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --visual_vars camera --seed 20262027 --p_seed 20262027 --visualize_attns

# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --ft_noise_std 5.0 0.15 --seed 20262027 --p_seed 20262027 --visualize_attns

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --prop_noise_std 0.001 0.01 --seed 20262027 --p_seed 20262027 --visualize_attns

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --ft_noise_std 5.0 0.15 --prop_noise_std 0.001 0.01 --seed 20262027 --p_seed 20262027 --visualize_attns

# All eval vars (no swap)
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_training_set_wristviews --split 12 --n_rollouts 5 --video_path_folder eval/ablation_training_set_wristviews --video_skip 1 --camera_names frontview left_wristview --set_canonical --eval_vars --eval_no_swap --seed 20262027 --p_seed 20262027 --visualize_attns