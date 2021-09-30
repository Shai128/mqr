import numpy as np


# ------------------------------------------ main figures saving dirs ------------------------------------------


def get_dir_aux_results_dir(dataset_name, transformation, epoch, is_real, seed, tau, dropout=0., hs='[64, 64, 64]',
                            vae_loss='KL',
                            vae_z_dim=3, vae_mode='CVAE', method_name='directional', kl_mult=0.01, **kwargs):
    if is_real:
        datatype = 'real'
    else:
        datatype = 'syn'
    if 'identity' in str(transformation):
        transformation_name = str(transformation)
    else:
        transformation_name = f"{str(transformation)}_{get_vae_save_name_by_params(vae_loss, vae_z_dim, vae_mode, kl_mult)}"
    model_params = f"method={method_name}_dropout={np.round(dropout, 2)}_hs={str(hs)}"
    base_save_dir = f"{datatype}/{dataset_name}/{transformation_name}_{model_params}_tau={tau}"
    return base_save_dir


def get_save_final_results_dir(dataset_name, **kwargs):
    base_save_dir = f"final_results/{get_dir_aux_results_dir(dataset_name=dataset_name, **kwargs)}"
    return base_save_dir


def get_save_final_figure_results_dir(dataset_name, **kwargs):
    base_save_dir = f"results/full_figures/{get_dir_aux_results_dir(dataset_name=dataset_name, **kwargs)}"
    return base_save_dir


def get_model_summary_save_dir(dataset_name, **kwargs):
    save_dir = f"results/summary_figures/{get_dir_aux_results_dir(dataset_name=dataset_name, **kwargs)}"
    return save_dir


# ------------------------------------------ VAE figures saving dirs ------------------------------------------


def get_vae_save_name_by_params(vae_loss, vae_z_dim, vae_mode, kl_mult=0.01):
    return f"mode={vae_mode}_loss={vae_loss}_z_dim={vae_z_dim}_kl={kl_mult}"


def get_encoder_base_save_dir_aux(dataset_name, epoch, is_real, seed, z_dim, loss, mode, kl_mult=0.01):
    if is_real:
        datatype = 'real'
    else:
        datatype = 'syn'
    base_save_dir = f"{datatype}/{dataset_name}/vae/seed={seed}/{get_vae_save_name_by_params(loss, z_dim, mode, kl_mult)}"
    return base_save_dir


def get_vae_results_save_dir(dataset_name, epoch, is_real, seed, z_dim, loss, mode, kl_mult=0.01, **kwargs):
    if is_real:
        datatype = 'real'
    else:
        datatype = 'syn'
    partial_save_dir = f"{datatype}/{dataset_name}/{get_vae_save_name_by_params(loss, z_dim, mode, kl_mult)}"
    return f'final_results/vae/{partial_save_dir}'


def get_vae_results_save_path(dataset_name, epoch, is_real, seed, z_dim, loss, mode, kl_mult=0.01, **kwargs):
    return f'{get_vae_results_save_dir(dataset_name, epoch, is_real, seed, z_dim, loss, mode, kl_mult=kl_mult)}/seed={seed}.csv'


def get_encoder_base_save_dir(**kwargs):
    base_save_dir = f"results/full_figures/{get_encoder_base_save_dir_aux(**kwargs)}"
    return base_save_dir


def get_encoder_summary_save_dir(**kwargs):
    base_save_dir = f"results/summary_figures/{get_encoder_base_save_dir_aux(**kwargs)}"
    return base_save_dir


# ------------------------------------------ VAE model saving dirs ------------------------------------------

def get_vae_model_save_path(dataset_name, seed, vae_loss, vae_z_dim, vae_mode, kl_mult=0.01):
    return f'models_save/{dataset_name}/seed={seed}/{get_vae_save_name_by_params(vae_loss, vae_z_dim, vae_mode, kl_mult)}'


def get_vae_model_save_name(dataset_name, seed, vae_loss, vae_z_dim, vae_mode, kl_mult=0.01):
    return f'{get_vae_model_save_path(dataset_name, seed, vae_loss, vae_z_dim, vae_mode, kl_mult)}/vae'


def get_cvae_model_save_name(dataset_name, seed, vae_loss, vae_z_dim, vae_mode, kl_mult=0.01):
    return f'{get_vae_model_save_path(dataset_name, seed, vae_loss, vae_z_dim, vae_mode, kl_mult)}/cvae'


# ------------------------------------------ Area plotting saving dir ------------------------------------------

def get_area_vs_d_figure_save_dir(is_linear, x_dim, tau):
    return f'results/area_figures/is_linear={is_linear}_x_dim={x_dim}_tau={tau}'


def get_area_vs_d_figure_save_path(is_linear, x_dim, scale, tau):
    return f'{get_area_vs_d_figure_save_dir(is_linear, x_dim, tau)}/scale={scale}.png'


# ------------------------------------------ VAE MSE losses plotting saving dir ------------------------------------------
def get_mse_vs_r_figure_save_dir():
    return 'results/vae_figures'


def get_mse_vs_r_figure_save_path():
    return f'{get_mse_vs_r_figure_save_dir()}/mse_vs_r.png'


# ------------------------------------------ Calibration example saving dir ------------------------------------------

def get_cal_example_save_dir():
    return f'results/cal_example'


def get_too_large_qr_figure_save_path():
    return f'{get_cal_example_save_dir()}/large_gamma.png'


def get_too_small_qr_figure_save_path():
    return f'{get_cal_example_save_dir()}/small_gamma.png'


def get_qr_complement_figure_save_path():
    return f'{get_cal_example_save_dir()}/out_of_qr.png'


def get_best_qr_figure_save_path():
    return f'{get_cal_example_save_dir()}/best_qr.png'


def get_case_1_too_large_qr_figure_save_path():
    return f'{get_cal_example_save_dir()}/case_1_large_gamma.png'


def get_case_1_too_small_qr_figure_save_path():
    return f'{get_cal_example_save_dir()}/case_1_small_gamma.png'


def get_case_1_best_gamma_figure_save_path():
    return f'{get_cal_example_save_dir()}/case_1_best_gamma.png'


def get_cov_vs_gamma__graph_figure_save_path():
    return f'{get_cal_example_save_dir()}/cov_vs_gamma_graph.png'

def get_shrinking_qr1_figure_save_path():
    return f'{get_cal_example_save_dir()}/shrinking_qr1.png'

def get_shrinking_qr2_figure_save_path():
    return f'{get_cal_example_save_dir()}/shrinking_qr2.png'


# ------------------------------------------ Grid size saving dir ------------------------------------------

def get_grid_info_save_path(dataset_name, seed):
    return f'{get_grid_info_save_dir(dataset_name, seed)}/grid_info.csv'


def get_grid_info_save_dir(dataset_name, seed):
    return f'grid_info/{dataset_name}/{seed}'


def get_encoder_transformed_y_fig_name():
    return 'transformed_y'


def get_original_and_reconstructed_y_fig_name():
    return 'original_and_reconstructed_y'


def get_save_halfspace_figure_sub_path():
    return 'directional_region'


def get_transformed_region_figure_name():
    return 'quantile_region.png'


def get_inverse_transformed_region_figure_name():
    return 'inverse_transformed_quantile_region.png'


def get_z_space_region_figure_name():
    return 'transformed_space_quantile_region.png'


def get_transformed_contour_figure_name():
    return 'quantile_contour.png'


def get_untransformed_region_figure_name():
    return 'untransformed_quantile_region.png'


def get_naive_quantile_contours_figure_name():
    return 'quantile_contour.png'


def get_vqr_results_dir(dataset_name, is_real, seed):
    ds_type = 'real_data' if is_real else 'syn_data'
    results_dir = f'VQR/results/{ds_type}/{dataset_name}/seed={seed}'
    return results_dir
