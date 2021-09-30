import re

import numpy as np
import torch as torch
from helper import set_seeds, y_grid_size_per_y_dim, z_grid_size_per_z_dim
from tqdm import tqdm
from utils.q_model_ens import MultivariateQuantileModel
from torch.utils.data import DataLoader, TensorDataset
from losses import multivariate_qr_loss
from helper import generate_directions
from plot_helper import evaluate_conditional_performance
import argparse
import os
import warnings
from datasets import datasets
from transformations import CVAETransform, ConditionalIdentityTransform
from directories_names import get_cvae_model_save_name, get_save_final_figure_results_dir, get_model_summary_save_dir, \
    get_save_final_results_dir

import ast

import matplotlib
from sys import platform

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(device)

def real_data_to_tau(dataset_name, transform):
    is_dqr = 'identity' in transform
    
    if is_dqr:
        tau = 0.05
        
    else:  # Our method
        assert 'vae' in transform
        
        if 'blog' in dataset_name:
            tau = 0.07
            
        elif 'bio' in dataset_name:
            tau = 0.05
            
        elif 'house' in dataset_name:
            tau = 0.05
            
        elif 'meps' in dataset_name:
            tau = 0.07
            
        else:
            raise ValueError("Invalid dataset")
    
    return tau
    
def synthetic_data_to_tau(dataset_name, transform):
    is_linear = 'nonlinear' not in dataset_name
    is_dqr = 'identity' in transform
    p = int(re.search(r'\d+', dataset_name).group())  # the dimension of x
    if 'quad' in dataset_name:
        d = 4
    elif 'triple' in dataset_name:
        d = 3
    else:
        d = 2
    
    if is_dqr:
        if is_linear and d == 2 and p == 10:
            tau = 0.05
        elif is_linear and d == 2 and p == 50:
            tau = 0.05
        elif is_linear and d == 2 and p == 100:
            tau = 0.05
        
        elif not is_linear and d == 2 and p == 1:
            tau = 0.05
            
        elif not is_linear and d == 3 and p == 1:
            tau = 0.05
        elif not is_linear and d == 3 and p == 10:
            tau = 0.05

        elif not is_linear and d == 4 and p == 1:
            tau = 0.02
        elif not is_linear and d == 4 and p == 10:
            tau = 0.02
            
        else:
            raise ValueError("Invalid dataset")
            
    else:  # Our method
        assert 'vae' in transform

        if is_linear and d == 2 and p == 10:
            tau = 0.05
        elif is_linear and d == 2 and p == 50:
            tau = 0.05
        elif is_linear and d == 2 and p == 100:
            tau = 0.05
        
        elif not is_linear and d == 2 and p == 1:
            tau = 0.07
            
        elif not is_linear and d == 3 and p == 1:
            tau = 0.07
        elif not is_linear and d == 3 and p == 10:
            tau = 0.07

        elif not is_linear and d == 4 and p == 1:
            tau = 0.03
        elif not is_linear and d == 4 and p == 10:
            tau = 0.05

        else:
            raise ValueError("Invalid dataset")

    return tau
            

def params_to_tau(dataset_name, ds_type, transform, cvae_z_dim):
    # if cvae_z_dim == 1:
    #     return 0.05

    transform = transform.lower()

    if 'syn' in ds_type.lower():
        tau = synthetic_data_to_tau(dataset_name, transform)
    else:
        tau = real_data_to_tau(dataset_name, transform)

    return tau


def parse_args_utils(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device
    args.num_ens = 1
    args.boot = 0
    args.hs = ast.literal_eval(args.hs)
    args.conformalization_tau = args.tau

    args.suppress_plots = False if args.suppress_plots == 0 else 1

    args.tau = params_to_tau(args.dataset_name, args.ds_type, args.transform, args.vae_z_dim)  # asking for slightly higher coverage
    args.tau_list = torch.Tensor([args.tau]).to(device)

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--tau', type=float, default=0.1,
                        help='quantile level')

    parser.add_argument('--dataset_name', type=str, default='banana',
                        help='dataset to use')

    parser.add_argument('--suppress_plots', type=int, default=0,
                        help='1 to disable all plots, or 0 to allow plots')

    parser.add_argument('--num_u', type=int, default=32,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=1000,
                        help='number of epochs')

    parser.add_argument('--hs', type=str, default="[64, 64, 64]",
                        help='hidden dimensions')

    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=100,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--ds_type', type=str, default="SYN",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of test set size')
    parser.add_argument('--calibration_ratio', type=float, default=0.4,  # 0.5 of training size
                        help='ratio of calibration set size')

    parser.add_argument('--save_training_results', type=int, default=0,
                        help='1 for saving results during training, or 0 for not saving')
    parser.add_argument('--transform', type=str, default="CVAE",
                        help='')

    parser.add_argument('--vae_loss', type=str, default="KL",
                        help="'KL' or 'MMD'")
    parser.add_argument('--vae_z_dim', type=int, default=3,
                        help="encoded dimension")
    parser.add_argument('--vae_mode', type=str, default='CVAE',
                        help="one of: CVAE | CVAE-GAN | CVAE-GAN-CLASS | Bicycle | Bicycle-CLASS")

    args = parser.parse_args()

    args = parse_args_utils(args)

    return args


if __name__ == '__main__':

    args = parse_args()
    dataset_name = args.dataset_name
    print("dataset_name: ", dataset_name, "transformation: ", args.transform,
          f"tau: {args.tau}, conformalization tau: {args.conformalization_tau}, seed={args.seed}")

    seed = args.seed
    set_seeds(seed)

    test_ratio = args.test_ratio
    calibration_ratio = args.calibration_ratio
    val_ratio = 0.2

    is_real = 'real' in args.ds_type.lower()
    scale = is_real
    data = datasets.get_split_data(dataset_name, is_real, device, test_ratio, val_ratio, calibration_ratio, seed, scale)
    x_train, x_val, y_train, y_val, x_test, y_te, = data['x_train'], data['x_val'], \
                                                    data['y_train'], data['y_val'], \
                                                    data['x_test'], data['y_te']
    scale_x = data['scale_x']
    scale_y = data['scale_y']
    x_dim = x_train.shape[1]

    if calibration_ratio > 0:
        x_cal, y_cal = data['x_cal'], data['y_cal']

    if args.transform == 'identity':
        transform = ConditionalIdentityTransform()
    elif args.transform == "VAE" or args.transform == "CVAE":
        transform = CVAETransform(
            get_cvae_model_save_name(dataset_name, seed, args.vae_loss, args.vae_z_dim, args.vae_mode), device=device)
    else:
        print("transform must be one of 'identity', 'VAE', 'CVAE")
        assert False

    untransformed_y_train = y_train
    y_train = transform.cond_transform(y_train, x_train)
    y_val = transform.cond_transform(y_val, x_val)

    dim_y = y_train.shape[1]

    y_grid_size = y_grid_size_per_y_dim[untransformed_y_train.shape[1]]
    z_grid_size = z_grid_size_per_z_dim[y_train.shape[1]]
    model_ens = MultivariateQuantileModel(input_size=x_dim, y_size=dim_y,
                                          hidden_dimensions=args.hs, dropout=args.dropout,
                                          lr=args.lr, wd=args.wd, num_ens=args.num_ens, device=args.device,
                                          y_grid_size=y_grid_size, z_grid_size=z_grid_size)

    # Data loader
    loader = DataLoader(TensorDataset(x_train, y_train),
                        shuffle=True,
                        batch_size=args.bs)

    # Loss function
    loss_fn = multivariate_qr_loss
    batch_loss = True
    assert len(args.tau_list) == 1
    eval_losses = []
    train_losses = []
    for ep in tqdm(range(args.num_ep)):

        if model_ens.done_training:
            break

        # Take train step
        ep_train_loss = []  # list of losses from each batch, for one epoch
        for batch in loader:
            u_list, gamma = generate_directions(dim_y, args.num_u, args.tau_list[0])
            args.gamma = gamma

            (xi, yi) = batch
            loss = model_ens.loss(loss_fn, xi, yi, u_list,
                                  batch_q=batch_loss,
                                  take_step=True, args=args)

            ep_train_loss.append(loss)

        ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0).item()
        train_losses += [ep_tr_loss]

        # Validation loss
        y_val = y_val.to(args.device)
        u_list, gamma = generate_directions(dim_y, args.num_u, args.tau_list[0])
        args.gamma = gamma

        ep_va_loss = model_ens.update_va_loss(
            loss_fn, x_val, y_val, u_list,
            batch_q=batch_loss, curr_ep=ep, num_wait=args.wait,
            args=args)
        eval_losses += [ep_va_loss.item()]

    params = {'dataset_name': dataset_name, 'transformation': transform, 'epoch': model_ens.best_va_ep[0],
              'is_real': is_real, 'seed': seed, 'tau': args.conformalization_tau,
              'vae_loss': args.vae_loss, 'vae_z_dim': args.vae_z_dim,
              'dropout': args.dropout, 'hs': str(args.hs), 'vae_mode': args.vae_mode}
    base_save_dir = get_save_final_figure_results_dir(**params)
    base_results_save_dir = get_save_final_results_dir(**params)
    summary_base_save_dir = get_model_summary_save_dir(**params)

    # evaluate_conditional_performance(model_ens, x_train, untransformed_y_train, y_train, x_test, y_te,
    #                                  base_save_dir, transform, is_conformalized=False, args=args,
    #                                  dataset_name=dataset_name, scale_x=scale_x, scale_y=scale_y,
    #                                  cache=None,
    #                                  summary_base_save_dir=summary_base_save_dir,
    #                                  base_results_save_dir=base_results_save_dir, is_real=is_real)

    if calibration_ratio > 0:
        model_ens.conformalize(x_cal, y_cal, untransformed_y_train, y_train, transform, args.conformalization_tau,
                               args.tau)
        evaluate_conditional_performance(model_ens, x_train, untransformed_y_train, y_train, x_test, y_te,
                                         base_save_dir, transform, is_conformalized=True, args=args,
                                         dataset_name=dataset_name, scale_x=scale_x, scale_y=scale_y, cache=None,
                                         summary_base_save_dir=summary_base_save_dir,
                                         base_results_save_dir=base_results_save_dir, is_real=is_real)
