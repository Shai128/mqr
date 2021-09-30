import matplotlib
import numpy as np
import torch as torch
from matplotlib import patches, pyplot as plt

import helper
from helper import create_folder_if_it_doesnt_exist, get_grid_borders_and_stride, get_grid
from directories_names import get_transformed_contour_figure_name
from plot_helper import get_conditional_data, float_to_str, plot_samples, \
    add_info_to_results_dict, save_final_performance_info, update_marginal_results


def evaluate_conditional_performance_aux(model, model_pred, X, Y, x_train, y_train,
                                         alpha, base_save_dir,
                                         title_begin,
                                         idx,
                                         results_dict,
                                         is_real,
                                         is_marginal,
                                         args):
    quantiles = torch.Tensor([alpha / 2, 1 - alpha / 2])

    if is_real or is_marginal:
        y_upper = model_pred[:, 1][idx]
        y_lower = model_pred[:, 0][idx]

    else:
        model_pred = model.predict_q(
            X[0, :].unsqueeze(0), quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None
        )
        y_upper = model_pred[:, 1]
        y_lower = model_pred[:, 0]

    for k in range(Y.shape[1]):
        k_variable_coverage_identifiers = (Y[:, k] <= y_upper[:, k]) & (Y[:, k] >= y_lower[:, k])
        k_variable_coverage = float_to_str(k_variable_coverage_identifiers.float().mean().item() * 100)
        results_dict[f'{title_begin} variable {k} coverage'] = [k_variable_coverage]

    coverage_identifiers = model.get_coverage_identifiers(Y, y_lower, y_upper)
    y_grid_size = model.y_grid_size ** (1 / y_train.shape[1])
    full_y_grid = get_grid(y_train, y_grid_size, y_train.shape[1], pad=0.2)
    border_max, border_min, stride = get_grid_borders_and_stride(y_train, y_grid_size, pad=0.2)
    covered_area = torch.ceil((y_upper - y_lower) / stride).prod(dim=1)

    coverage_res = {'coverages': coverage_identifiers,
                    'total_covered_area': covered_area}
    idx_for_helper = np.arange(len(coverage_identifiers)).astype(int) if is_real else None
    add_info_to_results_dict(results_dict, title_begin, coverage_res, idx=idx_for_helper, is_real=is_real)

    create_folder_if_it_doesnt_exist(base_save_dir)
    save_path = f'{base_save_dir}/{get_transformed_contour_figure_name()}'
    plot_quantile_region(Y, y_lower, y_upper, is_real, is_marginal, full_y_grid, args, title_begin,
                         save_fig_path=save_path)


def evaluate_performance(model, dataset_name, x_train, y_train, x_test, y_te, is_real, scale_x, scale_y, base_save_dir,
                         base_results_save_dir, summary_base_save_dir, is_conformalized, args):
    alpha = args.tau
    quantiles = torch.Tensor([alpha / 2, 1 - alpha / 2])

    with torch.no_grad():
        model_pred = model.predict_q(
            x_test, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None
        )

    title_begin = 'conformalized' if is_conformalized else 'unconformalized'

    coverages = {'x': [], 'coverage': [], 'area': []}
    results_dict = {}

    for x_id, X, Y, x_title, idx in get_conditional_data(dataset_name, x_train, x_test, y_te, is_real, scale_x,
                                                         scale_y):
        save_dir = f"{base_save_dir}/{title_begin}/{x_title}"
        create_folder_if_it_doesnt_exist(save_dir)

        curr_title_begin = f"{title_begin} model, {x_title}:"
        evaluate_conditional_performance_aux(model, model_pred, X, Y, x_train, y_train,
                                             alpha, save_dir,
                                             curr_title_begin,
                                             idx,
                                             results_dict,
                                             is_real,
                                             False,
                                             args)

        coverages['x'] += [x_id]
        coverages['coverage'] += [results_dict[f'{curr_title_begin} coverage'][0]]
        coverages['area'] += [results_dict[f'{curr_title_begin} area'][0]]

    curr_title_begin = f"{title_begin} model test set:"

    save_dir = f"{base_save_dir}/{title_begin}/marginal"
    evaluate_performance_params = {'model': model, 'model_pred': model_pred, 'X': x_test, 'Y': y_te, 'x_train': x_train,
                                   'y_train': y_train,
                                   'alpha': alpha, 'base_save_dir': save_dir,
                                   'title_begin': curr_title_begin,
                                   'idx': np.arange(0, x_test.shape[0]).astype(int),
                                   "results_dict": results_dict,
                                   "is_real": is_real,
                                   "is_marginal": True,
                                   'args': args}

    update_marginal_results(results_dict, coverages, base_save_dir, title_begin, curr_title_begin,
                            evaluate_conditional_performance_aux,
                            evaluate_conditional_performance_aux_params=evaluate_performance_params)

    save_final_performance_info(results_dict, coverages, summary_base_save_dir, base_results_save_dir, title_begin,
                                args)


n_clusters_plotted = 0


def plot_quantile_region(Y, y_lower, y_upper, is_real, is_marginal, full_y_grid, args, title_begin='',
                         save_fig_path=None):
    if args.suppress_plots or Y.shape[1] != 2:
        return
    global n_clusters_plotted

    if is_real or is_marginal:
        title = None  # title_begin + 'Quantile region'
        y_grid_repeated = full_y_grid.unsqueeze(1).repeat(1, len(y_upper), 1)
        in_region_idx = ((y_grid_repeated <= y_upper) & (y_grid_repeated >= y_lower)).float().prod(dim=-1).bool()
        in_region_points = y_grid_repeated[in_region_idx]
        idx = np.random.permutation(len(in_region_points))[:50000]
        in_region_points = in_region_points[idx]
        in_region_points = helper.filter_outlier_points(in_region_points)
        if n_clusters_plotted > 0:
            a_label = None
            b_label = None
        else:
            a_label = 'samples'
            b_label = 'quantile region'
        plot_samples(Y.cpu(), a_label, args, b=in_region_points.cpu(), b_label=b_label, a_color='b', axis_name='Y',
                     b_color='r', title=title, save_fig_path=save_fig_path, legend_place='lower left', b_alpha=0.015)
        if not is_marginal:
            n_clusters_plotted += 1
    else:
        title = None  # title_begin + 'Quantile contour'
        y_upper = y_upper.flatten().cpu()
        y_lower = y_lower.flatten().cpu()
        Y = Y.cpu()

        rect = patches.Rectangle((y_lower[0], y_lower[1]), (y_upper - y_lower)[0], (y_upper - y_lower)[1],
                                 linewidth=1, edgecolor='r', facecolor='r', label='quantile region', alpha=0.6)
        # Add the patch to the Axes
        figure = plt.figure()
        ax = figure.add_subplot()
        ax.add_patch(rect)
        ax.scatter(Y[:, 0], Y[:, 1], c='b', label='samples')
        plt.xlim([-2.5, 2.5])
        plt.ylim([0.25, 2.2])

        if n_clusters_plotted == 0:
            plt.legend('lower left')
            lines1 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='quantile region',
                                             markerfacecolor='red',
                                             markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
            lines2 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='samples',
                                             markerfacecolor='blue',
                                             markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
            legend = plt.legend(handles=[lines1, lines2], loc='lower left')

            for lh in legend.legendHandles:
                lh.set_alpha(1)

        plt.xlabel("Y0")
        plt.ylabel("Y1")
        if title is not None:
            plt.title(title)
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        n_clusters_plotted += 1
