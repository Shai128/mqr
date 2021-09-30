import os
import torch
import numpy as np
import random
from scipy.stats import ortho_group
import copy
import pandas as pd

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

results_path = './results/'
syn_data_path = './datasets/synthetic_data/'
dataset_base_path = "./datasets/real_data/"

y_grid_size_per_y_dim = {
    2: 3e3,
    3: 1e5,
    4: 2e5
}
z_grid_size_per_z_dim = {
    1: 1e3,
    2: 1e4,
    3: 4e4,
    4: 1e5
}


def read_csv(results_dir, result_param_names):
    result_params = {}
    for name in result_param_names:
        result_params[name] = torch.Tensor(pd.read_csv(f'{results_dir}/{name}.csv', index_col=0).values).to(device)
    return result_params


def generate_direction_aux(direction_shape, num_vectors, tau):
    if direction_shape == 1:
        assert num_vectors % 2 == 0
        u_list = torch.Tensor([tau] * (num_vectors // 2) + [-tau] * (num_vectors // 2)).unsqueeze(1).to(device)
        gamma = torch.Tensor([0] * num_vectors).to(device)
        idx = np.random.permutation(len(u_list))
        u_list = u_list[idx]
        return u_list, gamma

    unitaries = ortho_group.rvs(dim=direction_shape, size=num_vectors)
    unitaries = torch.Tensor(unitaries).to(device)
    u_list = unitaries[:, :, -1]
    gamma = unitaries[:, :, :-1]
    u_list = u_list * tau
    return u_list.to(device), gamma.to(device)


u_list0, gamma0 = None, None

USE_SAME_DIRECTIONS = True
MAX_DIRECTIONS = 2048


def get_base_directions(direction_shape, tau=1):
    u_list = []
    gamma_list = []
    for i in range(direction_shape):
        eye = torch.eye(direction_shape)
        u = eye[:, i]
        gamma = torch.cat([eye[:, :i], eye[:, i + 1:]], dim=1)
        u_list += [u]
        gamma_list += [gamma]
    if direction_shape == 2:
        u_list += [torch.Tensor([1 / np.sqrt(2), 1 / np.sqrt(2)])]
        gamma_list += [torch.Tensor([[-1 / np.sqrt(2)], [1 / np.sqrt(2)]])]
    u_list = torch.stack(u_list).to(device) * tau
    gamma_list = torch.stack(gamma_list).to(device)
    return u_list, gamma_list


def generate_directions(direction_shape, num_vectors, tau, return_idx=False):
    if not USE_SAME_DIRECTIONS:
        return generate_direction_aux(direction_shape, num_vectors, tau)
    else:
        global u_list0, gamma0
        if u_list0 is None:
            u_list0, gamma0 = generate_direction_aux(direction_shape, MAX_DIRECTIONS, tau)
            u_list0, gamma0 = u_list0, gamma0

        if num_vectors == MAX_DIRECTIONS:
            return copy.deepcopy(u_list0), copy.deepcopy(gamma0)

        idx = np.random.permutation(num_vectors)
        if return_idx:
            return copy.deepcopy(u_list0[idx]), copy.deepcopy(gamma0[idx]), idx
        else:
            return copy.deepcopy(u_list0[idx]), copy.deepcopy(gamma0[idx])


def sample_on_sphere(n, dim):
    radius = torch.rand(n)
    vec = torch.randn(n, dim)
    vec = ((vec.T / vec.norm(dim=1)) * radius).T

    return vec


def create_folder_if_it_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_current_seed():
    return torch.initial_seed()


def get_min_distance_for_points_per_y(y, points, idx=None):
    distances = torch.zeros(len(y))
    for i in range(len(y)):
        if idx is not None:
            curr_points = points[i][idx[i]]
        else:
            curr_points = points[i]
        distances[i] = get_min_distance(y[i].unsqueeze(0), curr_points)

    return distances


# no filtering!
def filter_outlier_points(sample, distance_quantile_level=0.999, neighbors_quantile_level=0.1):
    return sample

"""
        if len(sample) < 100:
            return sample
        distances = get_min_distance(sample, sample, ignore_zero_distance=True)  # distance from closest point in the sample
    
        # the distance such that most of the points have at least one neighbor in that distance
        q_distance = torch.quantile(distances, q=distance_quantile_level)
        n_neighbors = get_n_neighbors_in_radius(sample, sample, q_distance)  # number of neighbors in radius q_distance
    
        q_neighbors = n_neighbors.float().quantile(q=neighbors_quantile_level)
    
        filtered_samples = sample[n_neighbors >= q_neighbors]  # filter points that don't have enough neighbors
        return filtered_samples
"""


def get_n_neighbors_in_radius(y, points, radius):
    y_batch_size = 500

    points_batch_size = 5000
    total_n_neighbors = []
    for i in range(0, y.shape[0], y_batch_size):
        yi = y[i: min(i + y_batch_size, y.shape[0])]
        yi_n_neighbors_points = []
        for j in range(0, points.shape[0], points_batch_size):
            pts = points[j: min(j + points_batch_size, points.shape[0])]
            dist_from_pts = (yi - pts.unsqueeze(1).repeat(1, yi.shape[0], 1)).norm(dim=-1)
            n_neighbors = (dist_from_pts < radius).float().sum(dim=0)
            yi_n_neighbors_points += [n_neighbors]

        if len(yi_n_neighbors_points) > 0:
            total_n_neighbors += [torch.stack(yi_n_neighbors_points)]

    if len(total_n_neighbors) == 0:
        return torch.Tensor([0]).repeat(len(y)).to(y.device)
    else:
        return torch.cat(total_n_neighbors, dim=1).min(dim=0)[0]


def get_min_distance(y, points, ignore_zero_distance=False, y_batch_size=50, points_batch_size=10000):
    min_dists_from_points = []

    for i in range(0, y.shape[0], y_batch_size):
        yi = y[i: min(i + y_batch_size, y.shape[0])]
        yi_min_dists_from_points = []
        for j in range(0, points.shape[0], points_batch_size):
            pts = points[j: min(j + points_batch_size, points.shape[0])]
            dist_from_pts = (yi - pts.unsqueeze(1).repeat(1, yi.shape[0], 1)).norm(dim=-1)
            if ignore_zero_distance:
                dist_from_pts[dist_from_pts == 0] = np.inf
            min_dist_from_pts = dist_from_pts.min(dim=0)[0]
            yi_min_dists_from_points += [min_dist_from_pts]

        if len(yi_min_dists_from_points) > 0:
            min_dists_from_points += [torch.stack(yi_min_dists_from_points)]

    if len(min_dists_from_points) == 0:
        return torch.Tensor([np.inf]).repeat(len(y)).to(y.device)
    else:
        return torch.cat(min_dists_from_points, dim=1).min(dim=0)[0]


def get_grid_borders_and_stride(y_train, grid_size, pad=0.):
    q = 0.01
    border_min = y_train.quantile(q, dim=0)
    border_max = y_train.quantile(1 - q, dim=0)
    border_max += pad
    border_min -= pad

    stride = (border_max - border_min) / grid_size
    return border_max, border_min, stride


def get_grid_from_borders(border_max, border_min, stride, device):
    shifts = [torch.arange(
        border_min[i], border_max[i], step=stride[i], dtype=torch.float32, device=device
    ) for i in range(border_max.shape[0])]

    grid = torch.cartesian_prod(*shifts)
    return grid


def get_grid(y_train, grid_size, grid_shape, pad=0., get_grid_area=False):
    border_max, border_min, stride = get_grid_borders_and_stride(y_train, grid_size, pad)

    grid = get_grid_from_borders(border_max, border_min, stride, y_train.device)
    if len(grid.shape) == 1:
        grid = grid.unsqueeze(1)
    if get_grid_area:
        grid_area = (border_max - border_min).prod().item()
        return grid, grid_area
    else:
        return grid
