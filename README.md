## Harsh

python solution/train_pldm.py --config solution/configs/default_grid_config.yaml

## Kevin

python solution/train_pldm.py --config solution/configs/default_grid_config.yaml --data=data/raw/2020_hh_trials.csv
python -m solution.value.train_value data/raw/2020_hh_trials.csv discrete_sac --gpu
python -m solution.run_experiments data/raw/2020_hh_trials.csv data/output_dir --max_steps 500 --pldm_dir data/pldm --gpu --reward_model_path d3rlpy_logs/overcooked_discrete_sac_20250430193519