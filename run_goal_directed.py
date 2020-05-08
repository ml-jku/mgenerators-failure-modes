from copy import deepcopy

from optimize import optimize

opt_args = {}
opt_args['graph_ga'] = dict(
    smi_file='./data/guacamol_v1_train.smiles',
    population_size=100,
    offspring_size=200,
    generations=150,
    mutation_rate=0.01,
    n_jobs=-1,
    random_start=True,
    patience=150,
    canonicalize=False)

opt_args['lstm_hc'] = dict(
    pretrained_model_path='./guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt',
    n_epochs=151,
    mols_to_sample=1028,
    keep_top=512,
    optimize_n_epochs=1,
    max_len=100,
    optimize_batch_size=64,
    number_final_samples=1028,
    sample_final_model_only=False,
    random_start=True,
    smi_file='./data/guacamol_v1_train.smiles',
    n_jobs=-1,
    canonicalize=False)

# Set everything that varies in the loop to None
base_config = dict(
    chid=None,
    n_estimators=100,
    n_jobs=8,
    external_file='./data/guacamol_v1_test.smiles',
    n_external=3000,
    seed=None,
    opt_name=None,
    optimizer_args=None,
    log_base='results/goal_directed_v3')

n_runs = 10
for opt_name, optimizer_args in opt_args.items():
    for chid in ['CHEMBL1909203', 'CHEMBL1909140', 'CHEMBL3888429']:
        for i in range(0, n_runs):
            config = deepcopy(base_config)
            config['chid'] = chid
            config['seed'] = i
            config['opt_name'] = opt_name
            config['optimizer_args'] = optimizer_args

            print(f'Run {i+1}/{n_runs}, {opt_name}, {chid}')
            optimize(**config)
