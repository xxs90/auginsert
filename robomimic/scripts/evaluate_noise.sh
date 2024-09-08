# === 5% Noise === #
# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise5 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 5.0 0.15 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise5_evals

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise5 --video_skip 5 --camera_names overhead frontview --set_canonical --prop_noise_std 0.001 0.01 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise5_evals

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise5 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 5.0 0.15 --prop_noise_std 0.001 0.01 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise5_evals

# === 10% Noise === #
# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise10 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 10.0 0.30 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise10_evals

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise10 --video_skip 5 --camera_names overhead frontview --set_canonical --prop_noise_std 0.002 0.02 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise10_evals

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise10 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 10.0 0.30 --prop_noise_std 0.002 0.02 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise10_evals

# === 15% Noise === #
# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise15 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 15.0 0.45 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise15_evals

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise15 --video_skip 5 --camera_names overhead frontview --set_canonical --prop_noise_std 0.003 0.03 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise15_evals

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise15 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 15.0 0.45 --prop_noise_std 0.003 0.03 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise15_evals

# === 20% Noise === #
# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise20 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 20.0 0.60 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise20_evals

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise20 --video_skip 5 --camera_names overhead frontview --set_canonical --prop_noise_std 0.004 0.04 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise20_evals

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise20 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 20.0 0.60 --prop_noise_std 0.004 0.04 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise20_evals

# === Extreme Noise === #
# FT noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise100 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 100.0 1.50 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise100_evals

# Prop noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise100 --video_skip 5 --camera_names overhead frontview --set_canonical --prop_noise_std 0.01 0.1 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise100_evals

# Sensor noise
python robomimic/scripts/ctb_evaluate_all_policies.py --exp ablation_modality_input --split 3 --n_rollouts 50 --video_path_folder eval/ablation_noise100 --video_skip 5 --camera_names overhead frontview --set_canonical --ft_noise_std 100.0 1.50 --prop_noise_std 0.01 0.1 --seed 20262027 --p_seed 20262027 --wandb_proj_name ablation_noise100_evals