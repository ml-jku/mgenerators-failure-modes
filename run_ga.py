from optimize import optimize

config = dict(
    chid='CHEMBL1909203',
    n_estimators=100,
    n_jobs=8,
    external_file='./data/guacamol_v1_test.smiles.can',
    n_external=3000,
    seed=101,
    optimizer_args=dict(smi_file='./data/guacamol_v1_train.smiles.can',
                        population_size=100,
                        offspring_size=200,
                        generations=150,
                        mutation_rate=0.01,
                        n_jobs=-1,
                        random_start=True,
                        patience=5,
                        canonicalize=False))

n_runs = 10

for chid in ['CHEMBL1909203', 'CHEMBL1909140', 'CHEMBL1909140']:
    config['chid'] = chid
    for i in range(0, n_runs):
        print(f'Run {i+1}/{n_runs}')
        config['seed'] = i
        optimize(**config)