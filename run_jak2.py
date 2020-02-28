from optimize import optimize

config = dict(
    chid='CHEMBL3888429',
    n_estimators=100,
    n_jobs=8,
    external_file='./data/guacamol_v1_test.smiles.can',
    n_external=3000,
    seed=101,
    optimizer_args=dict(smi_file='./data/guacamol_v1_valid.smiles.can',
                        population_size=100,
                        offspring_size=200,
                        generations=150,
                        mutation_rate=0.01,
                        n_jobs=-1,
                        random_start=True,
                        patience=5,
                        canonicalize=False))

n_runs = 10
for i in range(3, n_runs):
    print(f'Run {i}/{n_runs}')
    config['seed'] = i
    optimize(**config)