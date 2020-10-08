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

opt_args['mso'] = dict(
    smi_file='./data/guacamol_v1_valid.smiles',
    num_part=200,
    num_iter=150)

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
    log_base='results/test3')


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from parametersearch import ParameterSearch
    import os

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, help='host address', default="localhost")
    parser.add_argument("--port", type=int, help='host port', default="7532")
    parser.add_argument("--server", help="run as client process", action="store_true")
    parser.add_argument("--work", help="run as client process", action="store_true")
    parser.add_argument("--nruns", help='How many runs to perform per task', default=10)
    args = parser.parse_args()

    if args.server == args.work:
        raise ValueError('Must be server or client only.')

    # Create a search grid and provide access to it on the network
    parameter_search = ParameterSearch(output_file=os.path.join(base_config['log_base'], 'results.csv'))

    if args.server:
        for opt_name in ['graph_ga', 'lstm_hc', 'mso']:
            optimizer_args = opt_args[opt_name]
            for chid in ['CHEMBL1909203', 'CHEMBL1909140', 'CHEMBL3888429']:
                for i in range(0, args.nruns):
                    config = deepcopy(base_config)
                    config['chid'] = chid
                    config['seed'] = i
                    config['opt_name'] = opt_name
                    config['optimizer_args'] = optimizer_args

                    # print(f'Run {i+1}/{args.nruns}, {opt_name}, {chid}')
                    parameter_search.add_parameter_setting(config)

        parameter_search.start_server('localhost', 7532, as_thread=False)

    elif args.work:
        parameter_search = ParameterSearch('localhost', 7532)
        for job_id, config in parameter_search:
            optimize(**config)
            continue
            try:
                optimize(**config)
                # report success
                parameter_search.submit_result(job_id, 0)
            except BaseException as e:
                # report failure
                parameter_search.submit_result(job_id, 1)
                print(e)
                if isinstance(e, KeyboardInterrupt):
                    print('Interrupted by user')
                    exit(1)
