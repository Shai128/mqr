import torch
from helper import create_folder_if_it_doesnt_exist, filter_outlier_points, get_grid_borders_and_stride, get_grid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from quantile_region import n_directions
from transformations import ConditionalIdentityTransform
from sklearn.manifold import TSNE
import six
from directories_names import get_inverse_transformed_region_figure_name, \
    get_z_space_region_figure_name, \
    get_untransformed_region_figure_name
import pandas as pd
import datasets
import math
from k_means_constrained import KMeansConstrained
import utils.q_model_ens
from functools import reduce
import operator

samples_alpha = 1.

import matplotlib as mpl

mpl.rcParams["legend.framealpha"] = 1
matplotlib.rc('font', **{'size': 15})


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def float_to_str(n):
    if type(n) == str:
        return n

    if abs(n) < 1e-3 and abs(n) > 1e-20 and not math.isnan(n):
        if n > 0:
            sign = ''
        else:
            sign = '-'
        return sign + f'{str(n).replace(".", "").replace("0", "")[0]}e' + str(int(np.log10(abs(n))))
    else:

        if (type(n) == float and n.is_integer()) or type(n) == int:
            n_str = str(int(n))
        else:
            n_str = str(np.round(n, 3))

        return n_str


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def get_conditional_data(dataset_name, x_train, x_test, y_te, is_real, scale_x, scale_y):
    device = x_train.device
    x_dim = x_test.shape[1]
    if is_real:
        n_clusters = 3

        x_cluster = torch.cat([x_test, y_te], dim=1)
        clustering_method = KMeansConstrained(n_clusters=n_clusters, size_min=int(len(x_cluster) * 0.2), random_state=0)
        # clustering_method = AgglomerativeClustering(n_clusters=n_clusters)
        clustering_method = clustering_method.fit(x_cluster.cpu().numpy())
        labels = clustering_method.labels_
        actual_n_clusters = labels.max() + 1

        for cluster in range(actual_n_clusters):
            cluster_idx = labels == cluster
            cluster_X = x_test[cluster_idx]
            cluster_Y = y_te[cluster_idx]
            x_title = f'cluster={cluster}'

            yield cluster, cluster_X, cluster_Y, x_title, cluster_idx

    else:
        for x in [1.5, 2, 2.5]:
            n = 4000
            X = torch.Tensor([[x]]).repeat(1, x_dim)
            Y = datasets.datasets.get_cond_dataset(dataset_name, n, X)
            X = X.repeat(n, 1)
            X = scale_x(X).to(device)
            Y = scale_y(Y).to(device)
            x_title = f'x={x}'

            yield x, X, Y, x_title, None


def plot_transformed_quantile_region(model, x_test, untransformed_y_test, transform, tau, args,
                                     n_directions=n_directions, title_begin='', base_save_dir=None,
                                     use_best_epoch=True):
    if args.suppress_plots or type(model) == utils.q_model_ens.VectorQuantileRegression:
        return

    if len(x_test) == 1:
        x_test_rep = x_test.repeat(len(untransformed_y_test), 1)
    else:
        x_test_rep = x_test
    transformed_y = transform.cond_transform(untransformed_y_test, x_test_rep)

    coverage_identifiers = model.is_in_region(x_test, transformed_y, tau,
                                              use_best_epoch=use_best_epoch, is_transformed=True, transform=transform,
                                              verbose=False, n_directions=n_directions).bool()

    covered_ys = untransformed_y_test[coverage_identifiers]
    uncovered_ys = untransformed_y_test[~coverage_identifiers]
    save_fig_path = f'{base_save_dir}/{get_untransformed_region_figure_name()}'
    plot_samples(covered_ys.cpu(), "covered y", b=uncovered_ys.cpu(), b_label='uncovered y',
                 title=title_begin + "$Y$ samples", axis_name='Y', a_color='r', b_color='b',
                 save_fig_path=save_fig_path,
                 a_alpha=.5, b_alpha=.5, args=args)


def add_info_to_results_dict(results_dict, title_begin, coverage_res, idx, is_real):
    marginal_coverages, total_covered_area = coverage_res['coverages'], coverage_res['total_covered_area']

    if idx is not None:
        curr_marginal_coverages = marginal_coverages[idx]
        curr_covered_area = total_covered_area[idx]
    else:
        curr_marginal_coverages = marginal_coverages
        curr_covered_area = total_covered_area

    coverage = curr_marginal_coverages.float().mean().item() * 100
    results_dict[f"{title_begin} coverage"] = [float_to_str(coverage)]
    results_dict[f"{title_begin} total area covered (%)"] = [float_to_str(curr_covered_area.mean().item())]
    results_dict[f'{title_begin} area'] = results_dict[f"{title_begin} total area covered (%)"]

    if is_real and coverage > 0:
        covered_samples_idx = curr_marginal_coverages == True
        results_dict[f"{title_begin} covered samples area covered (%)"] = [
            float_to_str(curr_covered_area[covered_samples_idx].mean().item())]
        results_dict[f'{title_begin} area'] = results_dict[f"{title_begin} covered samples area covered (%)"]


def save_final_performance_info(results_dict, coverages, summary_base_save_dir, base_results_save_dir, title_begin,
                                args):
    results_info_save_dir = f"{base_results_save_dir}/{title_begin}"
    create_folder_if_it_doesnt_exist(results_info_save_dir)
    pd.DataFrame(results_dict, index=[args.seed]).to_csv(f"{results_info_save_dir}/seed={args.seed}.csv")

    if not args.suppress_plots:
        create_folder_if_it_doesnt_exist(f"{summary_base_save_dir}/{title_begin}")
        pd.DataFrame(coverages).to_csv(f"{summary_base_save_dir}/{title_begin}/coverages.csv")
        ax = render_mpl_table(pd.DataFrame(coverages), header_columns=0, col_width=2.0)
        ax.set_title(f"{title_begin} performance")
        plt.savefig(f"{summary_base_save_dir}/{title_begin}/coverages.png", dpi=300)
        plt.show()


def update_marginal_results(results_dict, coverages, base_save_dir, title_begin, title,
                            evaluate_conditional_performance_aux,
                            evaluate_conditional_performance_aux_params):
    save_dir = f"{base_save_dir}/{title_begin}"
    create_folder_if_it_doesnt_exist(save_dir)
    curr_title_begin = title

    evaluate_conditional_performance_aux(**evaluate_conditional_performance_aux_params)

    marginal_coverage = results_dict[f'{curr_title_begin} coverage'][0]
    area = results_dict[f'{curr_title_begin} area'][0]

    coverages['x'] += ['marginal (test)']
    coverages['coverage'] += [float_to_str(marginal_coverage)]
    coverages['area'] += [float_to_str(area)]


def evaluate_conditional_performance(model, x_train, untransformed_y_train, transformed_y_train, x_test, y_te,
                                     base_save_dir, transform, is_conformalized, args, dataset_name,
                                     scale_x, scale_y, summary_base_save_dir, is_real, base_results_save_dir,
                                     cache=None):
    title_begin = 'conformalized' if is_conformalized else 'unconformalized'
    results_dict = {}
    z_dim = transformed_y_train.shape[1]

    _, _, y_stride = get_grid_borders_and_stride(untransformed_y_train,
                                                 model.y_grid_size ** (1 / untransformed_y_train.shape[1]),
                                                 pad=0.2)
    grid = get_grid(y_train=untransformed_y_train, grid_size=model.y_grid_size ** (1 / untransformed_y_train.shape[1]),
                    grid_shape=untransformed_y_train.shape[1], pad=0.2, get_grid_area=False)
    results_dict['n_cells_in_grid'] = grid.shape[0]

    coverage_res = model.get_coverage_identifiers(x_test, y_te, untransformed_y_train,
                                                  transformed_y_train, transform, args.tau,
                                                  cache=cache, get_quantile_region_sample=is_real)
    cache = coverage_res['cache']

    marginal_coverages, quantile_region_sample, quantile_out_region_sample, in_region_threshold = coverage_res[
                                                                                                      'coverages'], \
                                                                                                  coverage_res[
                                                                                                      'quantile_region_sample'], \
                                                                                                  coverage_res[
                                                                                                      'quantile_out_region_sample'], \
                                                                                                  coverage_res[
                                                                                                      'in_region_threshold']
    results_dict['radius'] = model.radius if is_conformalized else in_region_threshold.mean().item()
    if is_real:
        if model.is_conformalized:
            quantile_region_radius = model.radius
        else:
            quantile_region_radius = None

        def evaluate_conditional_performance_aux(model, x_test, untransformed_y_test, x_train,
                                                 untransformed_y_train, tau, base_save_dir,
                                                 title_begin, use_best_epoch, transform, idx):

            plot_transformed_quantile_region(model, x_test, untransformed_y_test, transform, tau,
                                             n_directions=n_directions, title_begin=title_begin,
                                             base_save_dir=base_save_dir, args=args)

            ys_in_region = torch.cat([quantile_region_sample['y'][i] for i in idx.nonzero()[0]], dim=0)
            zs_in_region = torch.cat([quantile_region_sample['z'][i] for i in idx.nonzero()[0]], dim=0)
            transformed_y = transform.cond_transform(untransformed_y_test, x_test)
            ys_idx = np.random.permutation(len(ys_in_region))[:10000]
            ys_in_region = ys_in_region[ys_idx]

            plot_inverse_transform_xi_results(xi=None, untransformed_y=untransformed_y_test,
                                              transformed_y=transformed_y,
                                              zs_in_region=zs_in_region, base_save_dir=base_save_dir,
                                              transform=transform,
                                              title_begin=title_begin, args=args,
                                              quantile_region_radius=quantile_region_radius, x_rep=x_test,
                                              ys_in_region=ys_in_region,
                                              filter_outliers=type(model) != utils.q_model_ens.VectorQuantileRegression)

            add_info_to_results_dict(results_dict, title_begin, coverage_res, idx, is_real=is_real)
            # coverage = results_dict[f"{title_begin} coverage"][0]
            # print(f"{title_begin} coverage: {coverage}%")

    else:
        def evaluate_conditional_performance_aux(model, x_test, untransformed_y_test, x_train,
                                                 untransformed_y_train, tau, base_save_dir,
                                                 title_begin, use_best_epoch, transform, idx):
            x_test = x_test[0]
            curr_coverage_res = model.get_coverage_identifiers(x_test, untransformed_y_test, untransformed_y_train,
                                                               transformed_y_train, transform, tau, cache=None,
                                                               get_quantile_region_sample=False)
            add_info_to_results_dict(results_dict, title_begin, curr_coverage_res, idx, is_real=is_real)
            syn_data_plot_quantile_region(model=model, x_test=x_test, untransformed_y_test=untransformed_y_test,
                                          x_train=x_train,
                                          untransformed_y_train=untransformed_y_train, tau=tau,
                                          base_save_dir=base_save_dir,
                                          args=args, title_begin=title_begin, use_best_epoch=use_best_epoch,
                                          transform=transform,
                                          z_dim=z_dim)

    coverages = {'x': [], 'coverage': [], 'area': []}
    for x_id, X, Y, x_title, idx in get_conditional_data(dataset_name, x_train, x_test, y_te, is_real, scale_x,
                                                         scale_y):
        save_dir = f"{base_save_dir}/{title_begin}/{x_title}"
        create_folder_if_it_doesnt_exist(save_dir)
        curr_title_begin = f"{title_begin} model, {x_title}:"

        evaluate_conditional_performance_aux(model, X, Y, x_train,
                                             untransformed_y_train, args.tau_list[0], save_dir,
                                             title_begin=curr_title_begin,
                                             use_best_epoch=True,
                                             transform=transform, idx=idx)
        coverages['x'] += [x_id]
        coverages['coverage'] += [results_dict[f"{curr_title_begin} coverage"][0]]
        coverages['area'] += [results_dict[f'{curr_title_begin} area'][0]]

    marginal_samples_idx = np.arange(len(y_te))
    curr_title_begin = f"{title_begin} model test set:"

    if is_real:
        save_dir = f"{base_save_dir}/{title_begin}/marginal"
        create_folder_if_it_doesnt_exist(save_dir)
        evaluate_performance_func = evaluate_conditional_performance_aux

        evaluate_performance_params = {'model': model, 'x_test': x_test, 'untransformed_y_test': y_te,
                                       'x_train': x_train,
                                       'untransformed_y_train': untransformed_y_train, 'tau': args.tau_list[0],
                                       'base_save_dir': save_dir, 'use_best_epoch': True, 'transform': transform,
                                       'idx': marginal_samples_idx}

    else:
        evaluate_performance_func = add_info_to_results_dict
        evaluate_performance_params = {'results_dict': results_dict,
                                       'coverage_res': coverage_res, 'is_real': is_real}

    evaluate_performance_params.update({
        'title_begin': curr_title_begin,
        'idx': marginal_samples_idx,
    })

    update_marginal_results(results_dict, coverages, base_save_dir, title_begin, curr_title_begin,
                            evaluate_performance_func,
                            evaluate_performance_params)

    save_final_performance_info(results_dict, coverages, summary_base_save_dir, base_results_save_dir, title_begin,
                                args)

    return cache


def plot_inverse_transform_results(model, x_test, untransformed_y, transformed_y_train, grid_size,
                                   base_save_dir, transform, tau, use_best_epoch, title_begin, args, filter_outliers,
                                   **kwargs):
    if args.suppress_plots:
        return

    transformed_dim = transformed_y_train.shape[1]
    possible_zs = get_grid(transformed_y_train, grid_size, transformed_dim, pad=0.2)

    if model.is_conformalized:
        quantile_region_radius = model.radius
    else:
        quantile_region_radius = None

    xi = x_test[0]
    zs_in_region = model.get_in_region_points(x_test, tau, z_grid=possible_zs, use_best_epoch=use_best_epoch,
                                              is_transformed=True, transform=transform, verbose=False)
    x_rep = x_test.repeat(untransformed_y.shape[0], 1)
    transformed_y = transform.cond_transform(untransformed_y, x_rep)

    plot_inverse_transform_xi_results(xi, untransformed_y, transformed_y, zs_in_region, base_save_dir, transform,
                                      title_begin, args=args, quantile_region_radius=quantile_region_radius,
                                      filter_outliers=filter_outliers, **kwargs)


def plot_inverse_transform_xi_results(xi, untransformed_y, transformed_y, zs_in_region, base_save_dir, transform,
                                      title_begin, args, quantile_region_radius=None, x_rep=None, ys_in_region=None,
                                      filter_outliers=False, **untransformed_y_plot_kwargs):
    if args.suppress_plots:
        return

    zs_to_color = zs_in_region
    if x_rep is None:
        assert xi is not None
        x_rep = xi.unsqueeze(0).repeat(zs_to_color.shape[0], 1)
    else:
        assert xi is None
    # title_begin = title_begin + ' '
    title_begin = ''
    create_folder_if_it_doesnt_exist(base_save_dir)
    transformed_y_idx = np.random.permutation(len(transformed_y))[:2000]
    if len(zs_to_color) < 1e+4 or transformed_y.shape[1] <= 4:
        save_fig_path = f'{base_save_dir}/{get_z_space_region_figure_name()}'
        idx = np.random.permutation(len(zs_to_color))[:10000]

        plot_samples(zs_to_color.cpu()[idx], "$R_\mathcal{Z}(x_\mathrm{new})$",
                     title=title_begin + "$Z|X$ Quantile region", axis_name='Z', a_color='r',
                     a_alpha=0.4, save_fig_path=f'{base_save_dir}/{"test_quantile_region_in_z_space.png"}', args=args)

        plot_samples(transformed_y.cpu()[transformed_y_idx], "$z$ samples", b=zs_to_color.cpu()[idx],
                     b_label="$R_\mathcal{Z}(x)$",
                     title=title_begin + "$Z|X$ Quantile region", axis_name='Z', a_color='b',
                     b_color='r',
                     a_alpha=0.15, b_alpha=0.4, save_fig_path=save_fig_path, args=args)

        plot_samples(untransformed_y.cpu(), "$y$ samples",
                     title=title_begin + "$Y$ samples", axis_name='Y', a_color='b',
                     a_alpha=1, save_fig_path=f'{base_save_dir}/y_samples.png', args=args,
                     **untransformed_y_plot_kwargs)

        plot_samples(transformed_y.cpu()[transformed_y_idx], "z samples",
                     title=title_begin + "Transformed samples", axis_name='Z', a_color='b',
                     a_alpha=0.15, save_fig_path=f'{base_save_dir}/z samples.png', args=args)

    if ys_in_region is None:
        n_points_to_plot = min(50000, len(zs_to_color))
        idx = np.random.permutation(len(zs_to_color))[:n_points_to_plot]
        ys_in_region = transform.cond_inverse_transform(zs_to_color[idx], x_rep[idx])

    if filter_outliers:
        ys_in_region = filter_outlier_points(ys_in_region)

    ys_to_color = ys_in_region

    if quantile_region_radius is not None and quantile_region_radius < 0:
        quantile_region_radius = None

    transformed_save_fig_path = f'{base_save_dir}/{get_inverse_transformed_region_figure_name()}'

    idx = np.random.permutation(len(ys_to_color))[:10000]
    if quantile_region_radius is None:
        b_alpha = 0.05
    elif quantile_region_radius < 0.01:
        b_alpha = 0.2
    elif quantile_region_radius < 0.05:
        b_alpha = 0.05
    elif quantile_region_radius < 0.1:
        b_alpha = 0.03
    else:
        b_alpha = 0.01

    if len(ys_to_color) < 3e3:
        ys_to_color_to_display = ys_to_color[idx].cpu().unsqueeze(0).repeat(int(np.ceil((3e3) // len(ys_to_color))), 1,
                                                                            1).flatten(0, 1)
    else:
        ys_to_color_to_display = ys_to_color[idx].cpu()

    ys_label = "quantile region"
    title = title_begin + '$Y|X$ Quantile region'
    plot_samples(untransformed_y.cpu(), b=ys_to_color_to_display, title=None,
                 axis_name='Y', a_color='b', b_color='r', save_fig_path=transformed_save_fig_path,
                 a_alpha=samples_alpha, b_alpha=b_alpha, b_radius=quantile_region_radius, args=args,
                 **untransformed_y_plot_kwargs)

    plot_samples(ys_to_color_to_display, "$R_\mathcal{Y}(x_\mathrm{new})$",
                 title=title, axis_name='Y', a_color='r',
                 save_fig_path=f'{base_save_dir}/{"test_quantile_region_in_y_space.png"}',
                 a_alpha=b_alpha, a_radius=quantile_region_radius, args=args, **untransformed_y_plot_kwargs)


def syn_data_plot_quantile_region(model, x_test, untransformed_y_test, x_train, untransformed_y_train, tau,
                                  base_save_dir, args,
                                  title_begin="", use_best_epoch=True, transform=ConditionalIdentityTransform(),
                                  z_dim=None):
    if args.suppress_plots:
        return

    x_test_rep = x_test

    if len(x_test.shape) == 1:
        x_test = x_test.unsqueeze(0)
    if x_test.shape[0] == 1:
        x_test_rep = x_test.repeat(untransformed_y_test.shape[0], 1)

    if z_dim is None:
        transformed_y_test = transform.cond_transform(untransformed_y_test, x_test_rep)
        z_dim = transformed_y_test.shape[1]

    if z_dim <= 3:
        grid_size = (5e+5) ** (1 / z_dim)
    elif z_dim < 5:
        grid_size = (5e+6) ** (1 / z_dim)
    else:
        grid_size = (5e+7) ** (1 / z_dim)

    plot_transformed_quantile_region(model, x_test, untransformed_y_test, transform, tau,
                                     n_directions=n_directions, title_begin=title_begin, base_save_dir=base_save_dir,
                                     args=args)

    create_folder_if_it_doesnt_exist(base_save_dir)

    transformed_y_train = transform.cond_transform(untransformed_y_train, x_train)
    plot_inverse_transform_results(model, x_test, untransformed_y_test, transformed_y_train, grid_size,
                                   base_save_dir, transform, tau, use_best_epoch, title_begin, args=args,
                                   filter_outliers=False,
                                   x_lim=[-2.5, 2.5], y_lim=[0.25, 2.2], legend_place='lower left')


def plot_samples(a, a_label=None, args=None, b=None, b_label=None, title=None, axis_name='X', a_color='b', b_color='g',
                 save_fig_path=None, a_alpha=1., b_alpha=1.,
                 a_radius=None, b_radius=None, legend_place='best', x_lim=None, y_lim=None, z_lim=None, legend_color='w', font_size=17):
    if args is not None and args.suppress_plots:
        return
    figure = plt.figure()
    matplotlib.rc('font', **{'size': font_size})
    if b is not None and len(b) == 0:
        b = None
    if b is not None:
        assert a.shape[1] == b.shape[1]
    if a.shape[1] > 5:
        return
    if a.shape[1] > 3:
        idx1 = np.random.permutation(len(a))[:1000]
        a = TSNE(n_components=3).fit_transform(a[idx1])
        if b is not None:
            idx2 = np.random.permutation(len(b))[:1000]
            b = TSNE(n_components=3).fit_transform(b[idx2])

    plot_params = {}
    a_plot_params = {
        'color': a_color,
        'alpha': a_alpha,
        **plot_params
    }
    if a_label is not None:
        a_plot_params['label'] = a_label

    if b is not None:
        b_plot_params = {
            'color': b_color,
            'alpha': b_alpha,
            **plot_params
        }
        if b_label is not None:
            b_plot_params['label'] = b_label

    if a.shape[1] == 3:
        ax = figure.add_subplot(projection='3d')

        if a_radius is not None:
            plt_sphere(figure, ax, a.numpy(), a_radius, a_color, a_alpha)

        else:
            ax.scatter(a[:, 0], a[:, 1], zs=a[:, 2], **a_plot_params)
        a_legend = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=a_label,
                                           markerfacecolor=a_color,
                                           markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
        if b is not None:

            if b_radius is not None:
                plt_sphere(figure, ax, b.numpy(), b_radius, b_color, b_alpha)

            else:
                ax.scatter(b[:, 0], b[:, 1], zs=b[:, 2], **b_plot_params)
            b_legend = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=b_label,
                                               markerfacecolor=b_color,
                                               markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
        handles = []
        if a_label is not None:
            handles += [a_legend]
        if b_label is not None:
            handles += [b_legend]
        if len(handles) > 0:
            plt.legend(handles=handles, loc=legend_place)

        ax.set_xlabel(axis_name + '0')
        ax.set_ylabel(axis_name + '1')
        ax.set_zlabel(axis_name + '2')


    elif a.shape[1] == 2:
        ax = figure.add_subplot()

        if a_radius is not None:
            circles = [plt.Circle(point, radius=a_radius, linewidth=0) for point in zip(*a.split(1, dim=1))]
            c = matplotlib.collections.PatchCollection(circles, **a_plot_params)

            a_legend = matplotlib.lines.Line2D([0], [0], marker='o', color=legend_color, label=a_label,
                                               markerfacecolor=a_color,
                                               markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
            ax.add_collection(c)
        else:
            a_legend = ax.scatter(a[:, 0], a[:, 1], **a_plot_params)

        if a_label is not None:
            plt.legend(handles=[a_legend], loc=legend_place)

        if b is not None:

            if b_radius is not None:
                circles = [plt.Circle(point, radius=b_radius, linewidth=0) for point in zip(*b.split(1, dim=1))]
                c = matplotlib.collections.PatchCollection(circles, **b_plot_params)
                ax.add_collection(c)
            else:
                ax.scatter(b[:, 0], b[:, 1], **b_plot_params)
            b_legend = matplotlib.lines.Line2D([0], [0], marker='o', color=legend_color, label=b_label,
                                               markerfacecolor=b_color,
                                               markersize=matplotlib.rcParams['lines.markersize'] * 1.3)

            handles = []
            if a_label is not None:
                handles += [a_legend]
            if b_label is not None:
                handles += [b_legend]
            if len(handles) > 0:
                plt.legend(handles=handles, loc=legend_place)

        ax.set_xlabel(axis_name + '0')
        ax.set_ylabel(axis_name + '1')

    else:
        ax = figure.add_subplot()
        ax.scatter(a[:, 0], np.zeros(len(a)), **a_plot_params)
        if b is not None:
            ax.scatter(b[:, 0], np.zeros(len(b)), **b_plot_params)
        plt.legend(loc=legend_place)
        ax.set_xlabel(axis_name)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if z_lim is not None:
        ax.set_zlim(z_lim)

    if title is not None:
        ax.set_title(title)
    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')

    plt.show()


def plt_sphere(fig, ax, list_center, radius, color, alpha):

    ax.scatter(list_center[:, 0], list_center[:, 1],list_center[:, 2], color=color, alpha=alpha)
