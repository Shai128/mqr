import re
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from directories_names import get_save_final_results_dir, get_grid_info_save_path, \
    get_area_vs_d_figure_save_path, get_area_vs_d_figure_save_dir, get_vae_results_save_dir, \
    get_mse_vs_r_figure_save_path, get_mse_vs_r_figure_save_dir
import pandas as pd

from helper import create_folder_if_it_doesnt_exist
from plot_helper import float_to_str

syn_data_x_values = [1.5, 2, 2.5]
clusters = 3
column_rename_map = {}
for conformalization_text in ['conformalized', 'unconformalized']:
    column_rename_map.update({
        f'{conformalization_text} model test set: coverage': 'coverage (%)',
        f'{conformalization_text} model test set coverage': 'coverage (%)',
        f'{conformalization_text} model test set: covered samples area covered (%)': 'covered samples area covered (%)',
        f'{conformalization_text} model test set covered samples area covered (%)': 'covered samples area covered (%)',
        f'{conformalization_text} model test set: total area covered (%)': 'all samples area covered (%)',
        f'{conformalization_text} model test set total area covered (%)': 'all samples area covered (%)',
    })
    for i in range(clusters):
        column_rename_map.update({
            f'{conformalization_text} model, cluster={i}: covered samples area covered (%)': f'cluster {i} area covered (%)',
            f'{conformalization_text} model, cluster={i}:  covered samples area covered (%)': f'cluster {i} area covered (%)',
            f'{conformalization_text} model, cluster={i}: coverage': f'cluster {i} coverage (%)',
            f'{conformalization_text} model, cluster={i}:  coverage': f'cluster {i} coverage (%)',
        })

    for x in syn_data_x_values:
        column_rename_map.update({
            f'{conformalization_text} model, x={x}: total area covered (%)': f'x={x} area covered (%)',
            f'{conformalization_text} model, x={x}:  total area covered (%)': f'x={x} area covered (%)',
            f'{conformalization_text} model, x={x}: coverage': f'x={x} coverage (%)',
            f'{conformalization_text} model, x={x}:  coverage': f'x={x} coverage (%)',
        })

column_rename_map.update({
    'x=marginal (test)_mse_loss': 'mse_loss',
    'x=marginal (test)_KL_loss': 'KL_loss'
})
area_column_name = 'all samples area covered (%)'
all_area_column_names = ['covered samples area covered (%)', 'all samples area covered (%)']
columns_to_present = ['coverage (%)', *all_area_column_names, 'radius', 'mse_loss', 'KL_loss']

for i in range(clusters):
    columns_to_present += [f'cluster {i} coverage (%)']
    columns_to_present += [f'cluster {i} area covered (%)']

for x in syn_data_x_values:
    columns_to_present += [f'x={x} coverage (%)']
    columns_to_present += [f'x={x} area covered (%)']

DEFAULT_R = 3


class Method:
    def __init__(self, method_name, plot_name, **method_args):
        self.method_name = method_name
        self.plot_name = plot_name
        self.method_args = method_args
        self.method_args['method_name'] = method_name

    def __str__(self):
        return f"{self.plot_name}, method_args: {self.method_args}"


class VAEDQR(Method):

    def __init__(self, r=DEFAULT_R, transformation='CVAE', **method_args):
        method_name = 'directional'
        plot_name = f'ST DQR r={r}'
        method_args['transformation'] = transformation
        method_args['vae_z_dim'] = r
        method_args['z_dim'] = r
        method_args['loss'] = 'KL'
        method_args['mode'] = 'CVAE'
        super().__init__(method_name, plot_name, **method_args)


class DQR(Method):

    def __init__(self, **method_args):
        method_name = 'directional'
        plot_name = f'NPDQR'
        method_args['transformation'] = 'cond_identity'
        super().__init__(method_name, plot_name, **method_args)
        self.method_args = method_args


class NaiveQR(Method):

    def __init__(self, **method_args):
        method_name = 'naive'
        plot_name = f'naive'
        method_args['transformation'] = 'cond_identity'
        super().__init__(method_name, plot_name, **method_args)


class VQR(Method):

    def __init__(self, **method_args):
        method_name = 'vector'
        plot_name = f'VQR'
        method_args['transformation'] = 'cond_identity'
        super().__init__(method_name, plot_name, **method_args)


possible_methods = []
for r in [1, 2, 3, 4]:
    possible_methods += [VAEDQR(r=r)]

possible_methods += [NaiveQR(), DQR(), VQR()]


def get_grid_size_df(dataset_name, seeds):
    grid_size_df = pd.DataFrame()
    for seed in range(seeds):
        try:
            df = pd.read_csv(f'{get_grid_info_save_path(dataset_name, seed)}', index_col=0)
        except Exception:
            print(f"missing grid size df for dataset: {dataset_name}, and seed: {seed}")
            break
        grid_size_df = grid_size_df.append(df)
    return grid_size_df


def read_one_method_results(dataset_name, is_real, seeds, tau, method, is_conformalized, reduction_method=np.mean,
                            scale_area_by_grid_size=False, result_to_read='qr'):
    all_dfs = pd.DataFrame()
    for seed in range(seeds):

        params = {'dataset_name': dataset_name, 'epoch': 0,
                  **method.method_args,
                  'is_real': is_real, 'seed': seed, 'tau': tau}

        if result_to_read == 'qr':
            base_results_save_dir = get_save_final_results_dir(**params)
            if is_conformalized:
                base_results_save_dir += '/conformalized'
            else:
                base_results_save_dir += '/unconformalized'

        elif result_to_read == 'vae':
            base_results_save_dir = get_vae_results_save_dir(**params)
        else:
            raise ValueError("result_to_read must be either 'qr' or 'vae'.")
        try:
            df = pd.read_csv(f'{base_results_save_dir}/seed={seed}.csv', index_col=0)

        except Exception:
            print(f"no results found for parameters: {params}")
            print()
            break
        all_dfs = all_dfs.append(df)
    all_dfs = all_dfs.rename(columns=column_rename_map)
    new_cols_order = [col for col in columns_to_present if col in all_dfs.columns]
    all_dfs = all_dfs[new_cols_order]
    if scale_area_by_grid_size and len(all_dfs) > 0:
        grid_size_df = get_grid_size_df(dataset_name, len(all_dfs))
        all_dfs[area_column_name] = all_dfs[area_column_name].values / grid_size_df.values
    all_dfs = all_dfs.apply(reduction_method, axis=0).to_frame(
        method.plot_name)

    return all_dfs


SCALE_BY_BEST_METHOD = 'scale area by best method'
SCALE_BY_GRID_SIZE = 'scale area by grid size'


def read_all_method_results(dataset_name, is_real, seeds, tau, is_conformalized, reduction_method=np.mean,
                            scale_area=SCALE_BY_BEST_METHOD, methods=possible_methods, result_to_read='qr'):
    total_df = pd.DataFrame()
    curr_possible_methods = deepcopy(methods)

    if 'reduced' not in dataset_name and is_real:
        curr_possible_methods = list(filter(lambda method: type(method) != VQR, curr_possible_methods))

    for method in curr_possible_methods:
        method_df = read_one_method_results(dataset_name, is_real, seeds, tau, method, is_conformalized,
                                            reduction_method=reduction_method,
                                            scale_area_by_grid_size=scale_area == SCALE_BY_GRID_SIZE,
                                            result_to_read=result_to_read)
        total_df = total_df.append(method_df.T)
    curr_all_area_column_names = [col for col in all_area_column_names if col in total_df.columns]
    if scale_area == SCALE_BY_BEST_METHOD:
        total_df[curr_all_area_column_names] /= total_df[curr_all_area_column_names].min(axis=0)

    total_df[np.isnan(total_df)] = '-'
    total_df = total_df.applymap(float_to_str)

    return total_df


ALL_REAL_DATASETS = ['bio', 'house', 'blog_data', 'meps_19', 'meps_20', 'meps_21']
ALL_SYN_DATASETS = ['cond_banana_k_dim_1', 'cond_banana_k_dim_10', 'cond_banana_k_dim_50', 'cond_banana_k_dim_100',
                    # 'cond_triple_banana_k_dim_1', 'cond_triple_banana_k_dim_10',
                    # 'cond_quad_banana_k_dim_1', 'cond_quad_banana_k_dim_10',

                    # 'cond_banana_k_dim_500','nonlinear_cond_banana_k_dim_10',

                    'nonlinear_cond_banana_k_dim_1',
                    'nonlinear_cond_triple_banana_k_dim_1', 'nonlinear_cond_triple_banana_k_dim_10',
                    'nonlinear_cond_quad_banana_k_dim_1', 'nonlinear_cond_quad_banana_k_dim_10'
                    ]


def standard_error_reduction(column):
    std_dev = np.std(column)
    sqrt_n = np.sqrt(len(column))
    return std_dev / sqrt_n


def display_dfs_given_type(dfs, result_to_read='qr'):
    dfs_names = ['Coverage', 'Area'] if result_to_read == 'qr' else ['MSE loss', 'KL Loss']
    for df, name in zip(dfs, dfs_names):
        print(name)
        display(df)


def read_all_real_data_results(tau, seeds=20, is_conformalized=True, metric_to_display='mean', datasets=None,
                               scale_area=None, methods=possible_methods, result_to_read='qr', **kwargs):
    if datasets is None:
        datasets = ALL_REAL_DATASETS
    df1, df2 = read_all_data_results_aux(True, datasets, tau, seeds, is_conformalized, metric_to_display,
                                         scale_area, methods=methods, result_to_read=result_to_read, **kwargs)
    display_dfs_given_type([df1, df2], result_to_read=result_to_read)


def get_syn_data_description_columns_by_name(dataset_names):
    dict = {'Setting': ['nonlinear' if 'nonlinear' in name else 'linear' for name in dataset_names],
            'd': [3 if 'triple' in name else 4 if 'quad' in name else 2 for name in dataset_names],
            'p': [int(re.search(r'\d+', name).group()) for name in dataset_names],
            }
    df = pd.DataFrame(dict)
    return df


def read_all_syn_data_results(tau, seeds=20, is_conformalized=True, metric_to_display='mean', datasets=None,
                              scale_area=None, return_dfs=False, display_dfs=True,result_to_read='qr', **kwargs):
    if datasets is None:
        datasets = ALL_SYN_DATASETS

    df1, df2 = read_all_data_results_aux(False, datasets, tau, seeds, is_conformalized, metric_to_display,
                                                     scale_area,result_to_read=result_to_read, **kwargs)
    descrtiptive_cols = get_syn_data_description_columns_by_name(df1.index.tolist())
    descrtiptive_cols.index = df1.index
    dfs = []
    for df in [df1, df2]:
        df = pd.concat([descrtiptive_cols, df], axis=1)
        df = df.set_index('Setting')
        df = df.sort_values(['Setting', 'd', 'p'])
        dfs += [df]

    if display_dfs:
        display_dfs_given_type(dfs, result_to_read=result_to_read)
        # display_coverage_and_area(coverage_df, area_df)

    if return_dfs:
        return tuple(dfs)


def str_to_float(n):
    if n == '-':
        return np.nan
    return float(n)


def display_mse_vs_r(tau=0.1, seeds=20, is_conformalized=True, metric_to_display='mean',
                     datasets=['meps_19'], r_values=[1, 2, 3, 4]):
    methods = [VAEDQR(r=r) for r in r_values]
    mse_loss, kl_loss = read_all_data_results_aux(True, datasets, tau, seeds, is_conformalized, metric_to_display,
                                         methods=methods, result_to_read='vae')
    for data in mse_loss.index:
        plt.plot(r_values, mse_loss.loc[data].to_frame(data).applymap(str_to_float).values, label=data,
                 linewidth=5)

    plt.xticks(r_values)
    plt.legend()
    plt.xlabel("Latent space dimension - $r$")
    plt.ylabel("MSE loss")
    # plt.title("CVAE MSE loss vs $r$")
    save_path = get_mse_vs_r_figure_save_path()
    save_dir = get_mse_vs_r_figure_save_dir()
    create_folder_if_it_doesnt_exist(save_dir)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


    plt.show()

def display_area_vs_y_dim(tau, seeds=20, is_conformalized=True, y_dims=[2, 3, 4], p=1, linear=False,
                          scale_area=SCALE_BY_GRID_SIZE):
    datasets = []
    linear_text = '' if linear else 'nonlinear_'
    if 2 in y_dims:
        datasets += [f'{linear_text}cond_banana_k_dim_{p}']
    if 3 in y_dims:
        datasets += [f'{linear_text}cond_triple_banana_k_dim_{p}']
    if 4 in y_dims:
        datasets += [f'{linear_text}cond_quad_banana_k_dim_{p}']
    coverage_df, area_df = read_all_data_results_aux(False, datasets, tau, seeds, is_conformalized,
                                                     metric_to_display='mean',
                                                     scale_area=scale_area,
                                                     methods=[VAEDQR(r=DEFAULT_R), NaiveQR(), DQR()])
    methods_name_map = {
        VAEDQR(r=DEFAULT_R).plot_name: 'ST DQR',
        NaiveQR().plot_name: 'Naive QR',
        DQR().plot_name: 'NPDQR',
    }
    for method in methods_name_map:
        plt.plot(y_dims, area_df[method].to_frame(method).applymap(str_to_float).values, label=methods_name_map[method],
                 linewidth=4)
    plt.xticks(y_dims)
    curr_y_ticks, _ = plt.yticks()
    y_ticks = [1] + list(curr_y_ticks[curr_y_ticks > 0])
    plt.yticks(y_ticks)
    plt.legend()
    plt.xlabel("The dimension of the response")
    plt.ylabel("Quantile region area")

    save_path = get_area_vs_d_figure_save_path(linear, p, scale_area, tau)
    save_dir = get_area_vs_d_figure_save_dir(linear, p, tau)
    create_folder_if_it_doesnt_exist(save_dir)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def read_all_data_results_aux(is_real, datasets, tau, seeds=20, is_conformalized=True, metric_to_display='mean',
                              scale_area=None, methods=possible_methods, result_to_read='qr', **kwargs):
    if metric_to_display == 'mean':
        reduction_method = np.mean
        scale_area = SCALE_BY_BEST_METHOD if scale_area is None else scale_area
    elif metric_to_display == 'std_err':
        reduction_method = standard_error_reduction
        scale_area = None if scale_area is None else scale_area
    else:
        assert False
    tab1_df = pd.DataFrame()
    tab2_df = pd.DataFrame()
    if result_to_read == 'qr':
        col1_name = 'coverage (%)'
        col2_name = area_column_name
    elif result_to_read == 'vae':
        col1_name = 'mse_loss'
        col2_name = 'KL_loss'
    else:
        raise ValueError("result_to_read must be either 'qr' or 'vae'.")
    for dataset in datasets:
        data_df = read_all_method_results(dataset, is_real, seeds, tau, is_conformalized,
                                          reduction_method=reduction_method, scale_area=scale_area, methods=methods,
                                          result_to_read=result_to_read, **kwargs)
        if len(data_df.index) == 0 or len(data_df.columns) == 0:
            continue
        tab1_df[dataset] = data_df[col1_name]
        tab2_df[dataset] = data_df[col2_name]

    tab1_df, tab2_df = tab1_df.T, tab2_df.T


    return tab1_df, tab2_df

