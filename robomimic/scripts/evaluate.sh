# Canonical
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Train vars
# python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --train_vars --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Eval grasp vars
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --obj_vars xt zt yr zr --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Eval peg/hole shape
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --obj_shape_vars arrow line pentagon hexagon diamond u --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Eval body shape
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --obj_body_shape_vars cube-thin cylinder-thin octagonal-thin --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Visual vars
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --visual_vars lighting texture arena-eval --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Camera vars
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --visual_vars camera --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --ft_noise_std 5.0 0.15 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --prop_noise_std 0.001 0.01 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --ft_noise_std 5.0 0.15 --prop_noise_std 0.001 0.01 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# peg/hole swap
# python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --var_swap --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# All eval vars
# python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --eval_vars --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals

# All eval vars (no swap)
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_num_latents_wristviews --split 12 --n_rollouts 50 --video_path_folder eval/ablation_num_latents_wristviews --video_skip 5 --camera_names frontview left_wristview --set_canonical --eval_vars --eval_no_swap --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_num_latents_wristviews_evals