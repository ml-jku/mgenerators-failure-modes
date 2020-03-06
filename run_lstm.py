from optimize_lstm import optimize

config = dict(
    chid=None,
    n_estimators=100,
    n_jobs=8,
    external_file='./data/guacamol_v1_test.smiles.can',
    n_external=3000,
    seed=101,
    optimizer_args = dict(pretrained_model_path = './guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt',
                            n_epochs = 150,
                            mols_to_sample = 1028,
                            keep_top = 512,
                            optimize_n_epochs = 1,
                            max_len = 100,
                            optimize_batch_size = 64,
                            number_final_samples = 1028,
                            sample_final_model_only = False,
                            random_start = True,
                            smi_file = './data/guacamol_v1_train.smiles.can',
                            n_jobs = -1,
                            canonicalize=False))

n_runs = 10
# chids = ['CHEMBL1909203', 'CHEMBL1909140', 'CHEMBL3888429']
chids = ['CHEMBL3888429']

for chid in chids:
    config['chid'] = chid
    for i in range(0, n_runs):
        print(f'Run {i+1}/{n_runs}')
        config['seed'] = i
        optimize(**config)