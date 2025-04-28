python -m solution.value.train_value data/raw/2020_hh_trials.csv nfq --gpu
python -m solution.value.train_value data/raw/2020_hh_trials.csv dqn --gpu
python -m solution.value.train_value data/raw/2020_hh_trials.csv double_dqn --gpu
python -m solution.value.train_value data/raw/2020_hh_trials.csv discrete_sac --gpu
python -m solution.value.train_value data/raw/2020_hh_trials.csv bcq --gpu
python -m solution.value.train_value data/raw/2020_hh_trials.csv cql --gpu