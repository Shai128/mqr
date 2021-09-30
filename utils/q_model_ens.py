"""
Code copied from https://github.com/YoungseogChung/calibrated-quantile-uq
"""
import math
import os, sys
import shutil
import time
from copy import deepcopy
import tqdm
import numpy as np
import torch
from scipy.stats import norm as norm_distr
from scipy.stats import t as t_distr
from scipy.interpolate import interp1d

import datasets.datasets
from directories_names import get_vqr_results_dir
from quantile_region import is_in_region
from transformations import ConditionalIdentityTransform, IdentityTransform
from helper import get_min_distance, get_grid_borders_and_stride, get_grid
import abc
from helper import create_folder_if_it_doesnt_exist, sample_on_sphere
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from NNKit.models.model import vanilla_nn
from scipy.optimize import linear_sum_assignment
from utils.utils import alpha_shape
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from VQR_helper import VQRTp
from helper import read_csv
import matplotlib.path as mplPath

"""
Define wrapper uq_model class
All uq models will import this class
"""


class uq_model(object):

    def predict(self):
        raise NotImplementedError('Abstract Method')


""" QModelEns Utils """


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def get_ens_pred_interp(unc_preds, taus, fidelity=10000):
    """
    unc_preds 3D ndarray (ens_size, 99, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    # taus = np.arange(0.01, 1, 0.01)
    y_min, y_max = np.min(unc_preds), np.max(unc_preds)
    y_grid = np.linspace(y_min, y_max, fidelity)
    new_quants = []
    avg_cdfs = []
    for x_idx in tqdm.tqdm(range(unc_preds.shape[-1])):
        x_cdf = []
        for ens_idx in range(unc_preds.shape[0]):
            xs, ys = [], []
            targets = unc_preds[ens_idx, :, x_idx]
            for idx in np.argsort(targets):
                if len(xs) != 0 and targets[idx] <= xs[-1]:
                    continue
                xs.append(targets[idx])
                ys.append(taus[idx])
            intr = interp1d(xs, ys,
                            kind='linear',
                            fill_value=([0], [1]),
                            bounds_error=False)
            x_cdf.append(intr(y_grid))
        x_cdf = np.asarray(x_cdf)
        avg_cdf = np.mean(x_cdf, axis=0)
        avg_cdfs.append(avg_cdf)
        t_idx = 0
        x_quants = []
        for idx in range(len(avg_cdf)):
            if t_idx >= len(taus):
                break
            if taus[t_idx] <= avg_cdf[idx]:
                x_quants.append(y_grid[idx])
                t_idx += 1
        while t_idx < len(taus):
            x_quants.append(y_grid[-1])
            t_idx += 1
        new_quants.append(x_quants)
    return np.asarray(new_quants).T


def get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95, score_distr='z'):
    """
    unc_preds 3D ndarray (ens_size, num_tau, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    num_ens, num_tau, num_x = unc_preds.shape
    len_tau = taus.size

    mean_pred = np.mean(unc_preds, axis=0)
    std_pred = np.std(unc_preds, axis=0, ddof=1)
    stderr_pred = std_pred / np.sqrt(num_ens)
    alpha = (1 - conf_level)  # is (1-C)

    # determine coefficient
    if score_distr == 'z':
        crit_value = norm_distr.ppf(1 - (0.5 * alpha))
    elif score_distr == 't':
        crit_value = t_distr.ppf(q=1 - (0.5 * alpha), df=(num_ens - 1))
    else:
        raise ValueError('score_distr must be one of z or t')

    gt_med = (taus > 0.5).reshape(-1, num_x)
    lt_med = ~gt_med
    assert gt_med.shape == mean_pred.shape == stderr_pred.shape
    out = (lt_med * (mean_pred - (float(crit_value) * stderr_pred)) +
           gt_med * (mean_pred + (float(crit_value) * stderr_pred))).T
    out = torch.from_numpy(out)
    return out


class QModelEns(uq_model):

    def __init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                 num_ens, device, output_size=1, nn_input_size=None):

        self.num_ens = num_ens
        self.device = device
        self.dropout = dropout
        # output_size  = 1
        if nn_input_size is None:
            nn_input_size = input_size + y_size

        self.model = [vanilla_nn(input_size=nn_input_size, output_size=output_size,
                                 hidden_dimensions=hidden_dimensions,
                                 dropout=dropout).to(device)
                      for _ in range(num_ens)]
        self.optimizers = [torch.optim.Adam(x.parameters(),
                                            lr=lr, weight_decay=wd)
                           for x in self.model]
        self.keep_training = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [None for _ in range(num_ens)]
        self.best_va_ep = [0 for _ in range(num_ens)]
        self.done_training = False
        self.is_conformalized = False

    def use_device(self, device):
        self.device = device
        for idx in range(len(self.best_va_model)):
            self.best_va_model[idx] = self.best_va_model[idx].to(device)

        if device.type == 'cuda':
            for idx in range(len(self.best_va_model)):
                assert next(self.best_va_model[idx].parameters()).is_cuda

    def print_device(self):
        device_list = []
        for idx in range(len(self.best_va_model)):
            if next(self.best_va_model[idx].parameters()).is_cuda:
                device_list.append('cuda')
            else:
                device_list.append('cpu')
        print(device_list)

    def loss(self, loss_fn, x, y, q_list, batch_q, take_step, args):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y, x, q_list, self.device, args)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx], y, x,
                                             q_list, self.device, args)
                ens_loss.append(loss.cpu().item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def loss_boot(self, loss_fn, x_list, y_list, q_list, batch_q, take_step, args):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y_list[idx], x_list[idx],
                                   q_list, self.device, args)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx],
                                             y_list[idx], x_list[idx], q_list,
                                             self.device, args)
                ens_loss.append(loss.detach().cpu().item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def update_va_loss(self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args):
        with torch.no_grad():
            va_loss = self.loss(loss_fn, x, y, q_list, batch_q, take_step=False, args=args)

        # if torch.isnan(va_loss):
        #     print("va loss is nan!")

        for idx in range(self.num_ens):
            if self.keep_training[idx]:
                if va_loss[idx] < self.best_va_loss[idx]:
                    self.best_va_loss[idx] = va_loss[idx]
                    self.best_va_ep[idx] = curr_ep
                    self.best_va_model[idx] = deepcopy(self.model[idx])
                else:
                    if curr_ep - self.best_va_ep[idx] > num_wait:
                        self.keep_training[idx] = False

        if not any(self.keep_training):
            self.done_training = True

        return va_loss

    #####
    def predict(self, cdf_in, conf_level=0.95, score_distr='z',
                recal_model=None, recal_type=None, use_best_va_model=True):
        """
        Only pass in cdf_in into model and return output
        If self is an ensemble, return a conservative output based on conf_bound
        specified by conf_level

        :param cdf_in: tensor [x, p], of size (num_x, dim_x + 1)
        :param conf_level: confidence level for ensemble prediction
        :param score_distr: 'z' or 't' for confidence bound coefficient
        :param recal_model:
        :param recal_type:
        :return:
        """

        if self.num_ens == 1:
            with torch.no_grad():
                if use_best_va_model:
                    pred = self.best_va_model[0](cdf_in)
                else:
                    pred = self.model[0](cdf_in)
        if self.num_ens > 1:
            pred_list = []
            if use_best_va_model:
                models = self.best_va_model
            else:
                models = self.model

            for m in models:
                with torch.no_grad():
                    pred_list.append(m(cdf_in).T.unsqueeze(0))

            unc_preds = torch.cat(pred_list, dim=0).detach().cpu().numpy()  # shape (num_ens, num_x, 1)
            taus = cdf_in[:, -1].flatten().cpu().numpy()
            pred = get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95,
                                           score_distr='z')
            pred = pred.to(cdf_in.device)

        return pred

    #####

    def predict_q(self, x, q_list=None, ens_pred_type='conf',
                  recal_model=None, recal_type=None, use_best_va_model=True):
        """
        Get output for given list of quantiles

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param ens_pred_type:
        :param recal_model:
        :param recal_type:
        :return:
        """

        if q_list is None:
            q_list = torch.arange(0.01, 0.99, 0.01)
        else:
            q_list = q_list.flatten()

        if self.num_ens > 1:
            # choose function to make ens predictions
            if ens_pred_type == 'conf':
                ens_pred_fn = get_ens_pred_conf_bound
            elif ens_pred_type == 'interp':
                ens_pred_fn = get_ens_pred_interp
            else:
                raise ValueError('ens_pred_type must be one of conf or interp')

        num_x = x.shape[0]
        num_q = q_list.shape[0]

        cdf_preds = []
        for p in q_list:
            if recal_model is not None:
                if recal_type == 'torch':
                    recal_model.cpu()  # keep recal model on cpu
                    with torch.no_grad():
                        in_p = recal_model(p.reshape(1, -1)).item()
                elif recal_type == 'sklearn':
                    in_p = float(recal_model.predict(p.flatten()))
                else:
                    raise ValueError('recal_type incorrect')
            else:
                in_p = float(p)
            p_tensor = (in_p * torch.ones(num_x)).reshape(-1, 1).to(x.device)

            cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
            cdf_pred = self.predict(cdf_in, use_best_va_model=use_best_va_model)  # shape (num_x, 1)
            cdf_preds.append(cdf_pred.unsqueeze(1))

        pred_mat = torch.cat(cdf_preds, dim=1)  # shape (num_x, num_q)
        # assert pred_mat.shape == (num_x, num_q)
        return pred_mat


class PredictQuantileRegion(abc.ABC):

    def __init__(self, y_grid_size, z_grid_size, **kwargs):
        super().__init__()
        self.is_conformalized = False
        self.radius = None
        self.y_grid_size = y_grid_size
        self.z_grid_size = z_grid_size

    @abc.abstractmethod
    def is_in_region(self, x, y, tau, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_in_region_points(self, x, tau, transform, z_grid, **kwargs):
        raise NotImplementedError

    # works for one sample only: x.shape = [1, d], where d is the feature dimension.
    def get_distance_from_quantile_region_aux(self, x, untransformed_y, in_region_threshold, z_grid, full_y_grid,
                                              transform, tau, update_quantile_region_sample, **kwargs):
        z_in_region = self.get_in_region_points(x=x, tau=tau, transform=transform, z_grid=z_grid, **kwargs)
        y_in_region = transform.cond_inverse_transform(z_in_region, x.repeat(len(z_in_region), 1))
        rnd_idx = np.random.permutation(len(y_in_region))[:20000]
        y_in_region = y_in_region[rnd_idx]

        if len(y_in_region) == 0:
            in_region_threshold = 0
        else:
            in_region_distances = get_min_distance(y_in_region, y_in_region, ignore_zero_distance=True,
                                                   y_batch_size=10000,
                                                   points_batch_size=10000)

            in_region_threshold = torch.quantile(in_region_distances, q=0.9).item()
        y_grid_dist_from_region = get_min_distance(full_y_grid, y_in_region, y_batch_size=20000,
                                                   points_batch_size=5000)

        out_of_region_idx = y_grid_dist_from_region > in_region_threshold
        y_out_region = full_y_grid[out_of_region_idx]
        n_out_of_region_out_of_entire_grid = len(y_out_region)
        rnd_idx = np.random.permutation(len(y_out_region))[:20000]
        y_out_region = y_out_region[rnd_idx]

        region_distance_as_outside_point = get_min_distance(untransformed_y, y_in_region)
        region_distance_as_inside_point = get_min_distance(untransformed_y, y_out_region)

        if self.radius is None:
            covered_area = (len(full_y_grid) - n_out_of_region_out_of_entire_grid)
        elif self.radius > 0:
            covered_area = (y_grid_dist_from_region < self.radius).float().sum()
        else:

            covered_area = (
                        get_min_distance(full_y_grid, y_out_region, y_batch_size=20000, points_batch_size=5000) > abs(
                    self.radius)).float().sum()


        update_quantile_region_sample(z_in_region, y_in_region, y_out_region)

        return region_distance_as_outside_point, region_distance_as_inside_point, covered_area, in_region_threshold

    def get_quantile_region_distance(self, x, untransformed_y, untransformed_y_train, transformed_y_train,
                                     transform, tau, get_quantile_region_sample=False):
        res = {}
        grid_step = self.z_grid_size ** (1 / transformed_y_train.shape[1])
        z_grid = get_grid(transformed_y_train, grid_step, transformed_y_train.shape[1], pad=1)
        device = x.device
        region_distance_as_outside_point = torch.zeros(len(untransformed_y)).to(device)
        region_distance_as_inside_point = torch.zeros(len(untransformed_y)).to(device)
        in_region_threshold = torch.zeros(len(untransformed_y)).to(device)
        total_covered_area = torch.zeros(len(untransformed_y)).to(device)

        y_grid_size = self.y_grid_size ** (1 / untransformed_y_train.shape[1])
        full_y_grid, y_grid_area = get_grid(untransformed_y_train, y_grid_size, untransformed_y_train.shape[1], pad=0.2,
                                            get_grid_area=True)

        _, _, y_stride = get_grid_borders_and_stride(untransformed_y_train, y_grid_size, pad=0.2)
        y_stride = y_stride.norm()

        if x.shape[0] == 1 and len(x.shape) == 2:
            x = x.flatten()

        def update_quantile_region_sample(z_in_region, y_in_region, y_out_region):
            return

        if len(x.shape) == 1:
            assert not get_quantile_region_sample
            region_distance_as_outside_point, region_distance_as_inside_point, covered_area, in_region_threshold = \
                self.get_distance_from_quantile_region_aux(x.unsqueeze(0), untransformed_y, y_stride, z_grid,
                                                           full_y_grid,
                                                           transform, tau, update_quantile_region_sample)
            total_covered_area = torch.ones(len(untransformed_y)).to(device) * covered_area

        else:
            if get_quantile_region_sample:
                quantile_region_sample = {'z': [], 'y': []}
                quantile_out_region_sample = {'z': [], 'y': []}
                sample_points_per_x = int(1e6 // len(x))
                res['quantile_region_sample'] = quantile_region_sample
                res['quantile_out_region_sample'] = quantile_out_region_sample

                def update_quantile_region_sample(z_in_region, y_in_region, y_out_region):
                    rnd_idx = np.random.permutation(len(z_in_region))[:sample_points_per_x]
                    nonlocal quantile_region_sample
                    quantile_region_sample['z'] += [z_in_region[rnd_idx]]

                    y_rnd_idx = np.random.permutation(len(y_in_region))[:sample_points_per_x]
                    quantile_region_sample['y'] += [y_in_region[y_rnd_idx]]

                    y_rnd_idx = np.random.permutation(len(y_out_region))[:sample_points_per_x]
                    quantile_out_region_sample['y'] += [y_out_region[y_rnd_idx]]

            for i in tqdm.tqdm(range(len(x))):
                region_distance_as_outside_point[i], region_distance_as_inside_point[i], total_covered_area[i], \
                in_region_threshold[i] = \
                    self.get_distance_from_quantile_region_aux(x[i].unsqueeze(0), untransformed_y[i].unsqueeze(0),
                                                               y_stride,
                                                               z_grid, full_y_grid,
                                                               transform, tau, update_quantile_region_sample)

        res['region_distance_as_outside_point'] = region_distance_as_outside_point
        res['region_distance_as_inside_point'] = region_distance_as_inside_point
        res['total_covered_area'] = total_covered_area
        res['in_region_threshold'] = in_region_threshold

        return res

    def conformalize(self, x_cal, untransformed_y_cal, untransformed_y_train, transformed_y_train,
                     transform, conformalization_tau, tau):
        qr_res = self.get_quantile_region_distance(x_cal, untransformed_y_cal, untransformed_y_train,
                                                   transformed_y_train, transform, tau)
        region_distance_as_outside_point, region_distance_as_inside_point, total_covered_area, in_region_threshold = \
        qr_res[
            'region_distance_as_outside_point'], \
        qr_res[
            'region_distance_as_inside_point'], \
        qr_res[
            'total_covered_area'], \
        qr_res[
            'in_region_threshold']

        is_in_qr = (region_distance_as_outside_point < in_region_threshold)  # is in quantile region
        print(f"calibration coverage before calibration: {np.round(is_in_qr.float().mean().item() * 100, 3)}%")

        n = len(untransformed_y_cal)
        q = np.ceil((n + 1) * (1 - conformalization_tau)) / n
        if is_in_qr.float().mean().item() <= 1 - conformalization_tau:  # we need to increase the quantile region radius
            scores = region_distance_as_outside_point
            self.radius = torch.quantile(scores, q=q).item()
        else:
            scores = region_distance_as_inside_point
            self.radius = torch.quantile(-scores, q=q).item()

        if math.isnan(self.radius):
            self.radius = None
        self.is_conformalized = True

        if self.radius > 0:
            cal_cov_identifiers = region_distance_as_outside_point < self.radius
        else:
            cal_cov_identifiers = region_distance_as_outside_point > abs(self.radius)

        print(
            f"calibration coverage after calibration: {np.round(cal_cov_identifiers.float().mean().item() * 100, 3)}%")
        print("radius: ", np.round(self.radius, 4))

    def get_coverage_identifiers(self, x_test, untransformed_y_test, untransformed_y_train,
                                 transformed_y_train, transform, tau, cache=None, get_quantile_region_sample=False):
        if cache is None:
            cache = self.get_quantile_region_distance(x_test, untransformed_y_test, untransformed_y_train,
                                                      transformed_y_train, transform, tau, get_quantile_region_sample)
        if get_quantile_region_sample:
            quantile_region_sample, quantile_out_region_sample = cache['quantile_region_sample'], cache[
                'quantile_out_region_sample']
        else:
            quantile_region_sample = quantile_out_region_sample = None

        region_distance_as_outside_point, region_distance_as_inside_point, total_covered_area, in_region_threshold = \
        cache[
            'region_distance_as_outside_point'], \
        cache[
            'region_distance_as_inside_point'], \
        cache[
            'total_covered_area'], \
        cache[
            'in_region_threshold']
        if not self.is_conformalized or self.radius is None:
            # _, _, radius = get_grid_borders_and_stride(untransformed_y_train,
            #                                            self.y_grid_size ** (1 / untransformed_y_train.shape[1]), pad=0.2)
            # radius = radius.norm().item()
            coverages = region_distance_as_outside_point < in_region_threshold
        else:
            radius = self.radius
            if radius > 0:
                coverages = region_distance_as_outside_point < radius  # distance from closest red point is at most radius
            else:
                coverages = region_distance_as_inside_point > abs(
                    radius)  # distance from the boundaries is at least radius

        res = {'coverages': coverages, 'cache': cache, 'quantile_region_sample': quantile_region_sample,
               'quantile_out_region_sample': quantile_out_region_sample, 'total_covered_area': total_covered_area,
               'in_region_threshold': in_region_threshold}

        return res


class VectorQuantileRegression(PredictQuantileRegion):

    def __init__(self, tau, device, y_grid_size=3e3, z_grid_size=4e4):
        PredictQuantileRegion.__init__(self, y_grid_size, z_grid_size)
        self.device = device
        self.tau = tau

    def fit(self, dataset_name, is_real, seed, x_train, y_train):
        self.y_dim = y_train.shape[1]
        data_dir = fr'VQR/Data/tmp/{dataset_name}/seed={seed}'
        create_folder_if_it_doesnt_exist(data_dir)
        pd.DataFrame(x_train.cpu().numpy()).to_csv(fr'{data_dir}/X.txt', header=None, index=None, sep='\t', mode='w+')
        pd.DataFrame(y_train.cpu().numpy()).to_csv(fr'{data_dir}/Y.txt', header=None, index=None, sep='\t', mode='w+')

        x_train = torch.cat([torch.ones(len(x_train)).unsqueeze(1).to(x_train.device), x_train], dim=1)
        results_dir = get_vqr_results_dir(dataset_name, is_real, seed)

        create_folder_if_it_doesnt_exist(results_dir)
        result_param_names = ['U'] + [f'beta{j + 1}' for j in range(self.y_dim)]
        device = self.device
        try:
            result_params = read_csv(results_dir, result_param_names)
        except Exception:  # in case the results do not exists
            VQRTp(x_train.cpu().numpy(), y_train.cpu().numpy(), results_dir)  # fit the model to get the results
            result_params = read_csv(results_dir, result_param_names)

        U = result_params['U']
        points = torch.Tensor(sample_on_sphere(len(U), 2)).to(device)
        cost_matrix = (U.unsqueeze(1).repeat(1, len(points), 1) - points.unsqueeze(0).repeat(len(U), 1, 1)).norm(
            dim=-1)  # .norm(dim=1)
        assign_sol = linear_sum_assignment(cost_matrix.cpu())
        assert (assign_sol[0] == np.array(list(range(len(U))))).all()
        points = points[assign_sol[1]]

        self.relevant_betas_idx = points.norm(dim=1) < 1 - self.tau

        self.betas = [None for _ in range(self.y_dim)]
        for j in range(self.y_dim):
            self.betas[j] = result_params[f'beta{j + 1}']
        # self.beta2 = result_params['beta2']

        shutil.rmtree(data_dir)

    def get_in_region_points(self, x, tau, transform, z_grid, **kwargs):
        # if len(x.shape) == 1 or x.shape[0] == 1:
        #     y_grid = z_grid
        #     coverage_identifiers = self.is_in_region(x, y_grid, tau, verbose=False)
        #     in_region_points = y_grid[coverage_identifiers]
        #     idx = np.random.permutation(len(in_region_points))[:1000]
        #     in_region_points = in_region_points[idx]
        # else:
        device = x.device
        x = x.flatten()
        curr_x_test = torch.cat([torch.ones(1).to(device), x], dim=0)
        Q = [(self.betas[j] @ curr_x_test).unsqueeze(1) for j in range(self.y_dim)]
        in_region_points = torch.cat(Q, dim=1)

        return in_region_points

    # check if the given points are inside the alpha-hull of the quantile region (inside quantile contour)
    def is_in_region(self, x, y, tau, verbose=True, **kwargs):
        device = x.device

        if len(x.shape) == 1 or x.shape[0] == 1:
            x = x.flatten()
            x = x.reshape(1, len(x))

        x = torch.cat([torch.ones(len(x)).unsqueeze(1).to(device), x], dim=1)

        Q = [torch.bmm(self.betas[j][self.relevant_betas_idx].unsqueeze(0).repeat(len(x), 1, 1),
                       x.unsqueeze(-1)).squeeze().unsqueeze(-1) for j in range(self.y_dim)]
        in_region_points = torch.cat(Q, dim=1).cpu()

        if x.shape[0] == 1:
            in_region_points = in_region_points.unsqueeze(0)
        coverage_identifiers = torch.zeros(len(y))

        x_range = range(len(x))
        if verbose:
            x_range = tqdm.tqdm(x_range)
        for i in x_range:
            curr_in_region_points = in_region_points[i]
            edges = list(alpha_shape(curr_in_region_points, 0.3))
            follower_dict = {}
            for e in edges:
                follower_dict[e[0]] = e[1]

            points_idx = []
            v = edges[0][0]
            while len(points_idx) - 1 != len(edges):
                points_idx += [v]
                v = follower_dict[v]
            curr_in_region_points = curr_in_region_points[points_idx]
            bbPath = mplPath.Path(curr_in_region_points)

            if len(x) == 1:
                coverage_identifiers = torch.Tensor(bbPath.contains_points(y.cpu()))
            else:
                coverage_identifiers[i] = bbPath.contains_points(y[i].cpu())

        return coverage_identifiers.bool()


class MultivariateQuantileModel(QModelEns, PredictQuantileRegion):

    def __init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                 num_ens, device, output_size=1, nn_input_size=None, y_grid_size=3e3, z_grid_size=4e4):
        QModelEns.__init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                           num_ens, device, output_size, nn_input_size)
        PredictQuantileRegion.__init__(self, y_grid_size, z_grid_size)

    def is_in_region(self, x, y, tau, use_best_epoch=True, is_transformed=True,
                     transform=ConditionalIdentityTransform(), verbose=False, **kwargs):
        if 'n_directions' not in kwargs:
            kwargs['n_directions'] = 256
        coverage_identifiers = is_in_region(self, x, y, tau, use_best_epoch=use_best_epoch,
                                            is_transformed=is_transformed, transform=transform, verbose=verbose,
                                            **kwargs)
        return coverage_identifiers

    def get_in_region_points(self, x, tau, transform, z_grid, **kwargs):
        coverage_identifier = self.is_in_region(x, z_grid, tau, use_best_epoch=True,
                                                is_transformed=True, transform=transform, verbose=False)
        return z_grid[coverage_identifier]


class NaiveMultivariateQuantileModel(QModelEns):

    def __init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                 num_ens, device, output_size=1, nn_input_size=None, y_grid_size=3e3, z_grid_size=4e4):
        QModelEns.__init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                           num_ens, device, output_size, nn_input_size)
        self.y_grid_size = y_grid_size
        self.z_grid_size = z_grid_size
        self.is_conformalized = False

    def get_coverage_identifiers(self, Y, y_lower, y_upper):
        return ((Y <= y_upper) & (Y >= y_lower)).float().prod(dim=1).bool()

    def conformalize(self, x_cal, y_cal, conformalization_tau, tau):

        quantiles = torch.Tensor([tau / 2, 1 - tau / 2])
        model_pred = self.predict_q(
            x_cal, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None
        )
        y_upper = model_pred[:, 1]
        y_lower = model_pred[:, 0]

        distance_from_boundaries = torch.zeros(len(y_cal), 2 * y_cal.shape[1]).to(y_cal.device)
        for dim in range(y_cal.shape[1]):
            distance_from_boundaries[:, dim * 2] = y_lower[:, dim] - y_cal[:, dim]
            distance_from_boundaries[:, dim * 2 + 1] = y_cal[:, dim] - y_upper[:, dim]
        scores = distance_from_boundaries.max(dim=1)[0]
        n = len(scores)
        q = np.ceil((n + 1) * (1 - conformalization_tau)) / n
        Q = torch.quantile(scores, q=q)
        self.correction = torch.ones(y_cal.shape[1]).to(y_cal.device) * Q
        self.radius = Q
        self.is_conformalized = True

    def predict_q(self, x, q_list=None, ens_pred_type='conf',
                  recal_model=None, recal_type=None, use_best_va_model=True):

        pred = super().predict_q(x, q_list, ens_pred_type, recal_model, recal_type, use_best_va_model)

        if self.is_conformalized:
            pred[:, 1] = pred[:, 1] + self.correction  # upper quantile
            pred[:, 0] = pred[:, 0] - self.correction  # lower quantile

        return pred
