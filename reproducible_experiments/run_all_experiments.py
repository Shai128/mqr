import subprocess
import sys
from itertools import product
import time
from run_experiment import run_experiment

# sys.path.insert(0, '../')


def cartesian_product(inp):
    if len(inp) == 0:
        return []
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


# programs = ['train_vae', 'main', 'naive_VQR', 'VQR']
real_datasets = ['meps_19', 'house', 'bio', 'blog_data', 'meps_20', 'meps_21']
reduced_real_datasets = ['reduced_' + data for data in real_datasets]
syn_datasets = ['cond_banana_k_dim_1', 'cond_banana_k_dim_10',
                'cond_banana_k_dim_50', 'cond_banana_k_dim_100',

                'nonlinear_cond_banana_k_dim_1',
                'nonlinear_cond_triple_banana_k_dim_1', 'nonlinear_cond_triple_banana_k_dim_10',
                'nonlinear_cond_quad_banana_k_dim_1', 'nonlinear_cond_quad_banana_k_dim_10'
                ]

processes_to_run_in_parallel = 1
programs_to_run = ['train_vae', 'main', 'naive_VQR', 'VQR']
seeds = list(range(20))  # [0]  # list(range(20))
suppress_plots = 1
real_datasets_to_run = real_datasets + reduced_real_datasets
syn_datasets_to_run = syn_datasets
run_real_data = True
run_syn_data = True
taus = [0.1]
fit_vqr_only = [0]
transform = ['CVAE', 'identity']  # ['CVAE', 'identity']
z_dim = [3]

syn_vae_params = {
    'main_program_name': ['train_vae'],
    'dataset_name': syn_datasets_to_run,
    'ds_type': ['SYN'],
    'loss': ['KL'],
    'z_dim': z_dim,
    'num_ep': [10000],
    'bs': [512],
    'lr': [1e-3],
    'mode': ['CVAE'],
    'seed': seeds,
    'suppress_plots': [suppress_plots],
    # 'mode': ['CVAE', 'CVAE-GAN', 'CVAE-GAN-CLASS', 'Bicycle', 'Bicycle-CLASS']
}

syn_main_vae_params = {
    'main_program_name': ['main'],
    'save_training_results': [0],
    'dataset_name': syn_datasets_to_run,
    "transform": ['CVAE'],
    'ds_type': ['SYN'],
    'tau': taus,
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],  # , '"[64, 128, 128, 64]"', '"[64, 128, 256, 128, 64]"'],
    # 'vae_loss': ['KL'],
    'vae_z_dim': z_dim,
    'vae_mode': ['CVAE'],
    'seed': seeds,
    'suppress_plots': [suppress_plots],

}

syn_main_dqr_params = {
    'main_program_name': ['main'],
    'save_training_results': [0],
    'dataset_name': syn_datasets_to_run,
    "transform": ['identity'],
    'ds_type': ['SYN'],
    'tau': taus,
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],  # , '"[64, 128, 128, 64]"', '"[64, 128, 256, 128, 64]"'],
    # 'vae_loss': ['KL'],
    'vae_mode': ['CVAE'],
    'seed': seeds,
    'suppress_plots': [suppress_plots],

}


syn_naive_qr_params = {
    'main_program_name': ['naive_VQR'],
    'dataset_name': syn_datasets_to_run,
    'ds_type': ['SYN'],
    'tau': taus,
    'seed': seeds,
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],
    'suppress_plots': [suppress_plots],
}
syn_vqr_params = {
    'main_program_name': ['VQR'],
    'fit_vqr_only': fit_vqr_only,
    'tau': taus,
    'seed': seeds,
    'dataset_name': syn_datasets_to_run,
    'ds_type': ['SYN'],
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],
    'suppress_plots': [suppress_plots],
}

real_vae_params = {
    'main_program_name': ['train_vae'],
    'dataset_name': real_datasets_to_run,
    'ds_type': ['REAL'],
    'seed': seeds,
    'loss': ['KL'],
    'z_dim': z_dim,
    'num_ep': [10000],
    'bs': [512],
    'lr': [1e-4],
    'mode': ['CVAE'],  # , 'CVAE-GAN', 'CVAE-GAN-CLASS', 'Bicycle', 'Bicycle-CLASS']
    'suppress_plots': [suppress_plots],

}

real_main_vae_params = {
    'main_program_name': ['main'],
    'save_training_results': [0],
    'dataset_name': real_datasets_to_run,
    "transform": ['CVAE'],
    'ds_type': ['REAL'],
    'tau': taus,
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],  # , '"[64, 128, 128, 64]"', '"[64, 128, 256, 128, 64]"'],
    # 'vae_loss': ['KL'],
    'vae_z_dim': z_dim,
    'vae_mode': ['CVAE'],
    'seed': seeds,
    'suppress_plots': [suppress_plots],

}

real_main_dqr_params = {
    'main_program_name': ['main'],
    'save_training_results': [0],
    'dataset_name': real_datasets_to_run,
    "transform": ['identity'],
    'ds_type': ['REAL'],
    'tau': taus,
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],  # , '"[64, 128, 128, 64]"', '"[64, 128, 256, 128, 64]"'],
    # 'vae_loss': ['KL'],
    'vae_mode': ['CVAE'],
    'seed': seeds,
    'suppress_plots': [suppress_plots],

}

real_naive_qr_params = {
    'main_program_name': ['naive_VQR'],
    'dataset_name': real_datasets_to_run,
    'ds_type': ['REAL'],
    'tau': taus,
    'seed': seeds,
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],
    'suppress_plots': [suppress_plots],  # , '"[64, 128, 128, 64]"', '"[64, 128, 256, 128, 64]"'],
}

real_vqr_params = {
    'main_program_name': ['VQR'],
    'fit_vqr_only': fit_vqr_only,
    'tau': taus,
    'seed': seeds,
    'dataset_name': real_datasets_to_run,
    'ds_type': ['REAL'],
    'bs': [256],
    'num_ep': [10000],
    'dropout': [0.],
    'hs': ['"[64, 64, 64]"'],
    'suppress_plots': [suppress_plots],  # , '"[64, 128, 128, 64]"', '"[64, 128, 256, 128, 64]"'],
}
params = []

if 'train_vae' in programs_to_run:
    if run_real_data:
        params += list(cartesian_product(real_vae_params))
    if run_syn_data:
        params += list(cartesian_product(syn_vae_params))

if 'main' in programs_to_run:
    if run_real_data:
        if 'identity' in transform:
            params += list(cartesian_product(real_main_dqr_params))
        if 'CVAE' in transform:
            params += list(cartesian_product(real_main_vae_params))
    if run_syn_data:
        if 'identity' in transform:
            params += list(cartesian_product(syn_main_dqr_params))
        if 'CVAE' in transform:
            params += list(cartesian_product(syn_main_vae_params))

if 'naive_VQR' in programs_to_run:
    if run_real_data:
        params += list(cartesian_product(real_naive_qr_params))
    if run_syn_data:
        params += list(cartesian_product(syn_naive_qr_params))

if 'VQR' in programs_to_run:
    if run_real_data:
        params += list(cartesian_product(real_vqr_params))
    if run_syn_data:
        params += list(cartesian_product(syn_vqr_params))

processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(params))

if __name__ == '__main__':

    print("jobs to do: ", len(params))
    # initializing proccesses_to_run_in_parallel workers
    workers = []
    jobs_finished_so_far = 0
    assert len(params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = params.pop(0)
        main_program_name = curr_params['main_program_name']
        curr_params.pop('main_program_name')
        p = run_experiment(curr_params, main_program_name)
        workers.append(p)

    # creating a new process when an old one dies
    while len(params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(params) > 0:
                curr_params = params.pop(0)
                main_program_name = curr_params['main_program_name']
                curr_params.pop('main_program_name')
                p = run_experiment(curr_params, main_program_name)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(params)} jobs left")
            time.sleep(10)

    # joining all last proccesses
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")
