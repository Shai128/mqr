import numpy as np
import torch as torch
from helper import set_seeds, y_grid_size_per_y_dim
from tqdm import tqdm
from utils.q_model_ens import NaiveMultivariateQuantileModel
from torch.utils.data import DataLoader, TensorDataset
from losses import naive_multivariate_qr_loss
import argparse
import matplotlib
import os
import warnings
from datasets import datasets
from transformations import ConditionalIdentityTransform
from directories_names import get_save_final_figure_results_dir, get_save_final_results_dir, get_model_summary_save_dir
import ast
from naive_VQR_helper import evaluate_performance

warnings.filterwarnings("ignore")

from sys import platform

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--tau', type=float, default=0.1,
                        help='quantile level')
    parser.add_argument('--suppress_plots', type=int, default=0,
                        help='1 to disable all plots, or 0 to allow plots')

    parser.add_argument('--dataset_name', type=str, default='banana',
                        help='dataset to use')

    parser.add_argument('--num_u', type=int, default=32,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=10000,
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

    parser.add_argument('--ds_type', type=str, default="REAL",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of test set size')
    parser.add_argument('--calibration_ratio', type=float, default=0.4,  # 0.5 of training size
                        help='ratio of calibration set size')

    parser.add_argument('--save_training_results', type=int, default=0,
                        help='1 for saving results during training, or 0 for not saving')
    parser.add_argument('--transform', type=str, default="identity",
                        help='')

    parser.add_argument('--vae_loss', type=str, default="KL",
                        help="'KL' or 'MMD'")
    parser.add_argument('--vae_z_dim', type=int, default=3,
                        help="encoded dimension")

    args = parser.parse_args()

    assert 'identity' in args.transform

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device
    args.num_ens = 1
    args.boot = 0
    args.hs = ast.literal_eval(args.hs)

    args.suppress_plots = False if args.suppress_plots == 0 else 1

    return args


if __name__ == '__main__':
    loss = 'pinball'
    TRAINING_OVER_ALL_QUANTILES = 'int' in loss
    args = parse_args()

    dataset_name = args.dataset_name
    device = args.device

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

    d = y_train.shape[1]
    tau_per_dimension = args.tau / d  # beta = 1 - alpha/d

    args.conformalization_tau = args.tau  # total desired coverage
    args.tau = tau_per_dimension
    args.tau_list = torch.Tensor([args.tau]).to(device)

    print("dataset_name: ", dataset_name, "transformation: ", args.transform,
          f"tau: {args.tau}, conformalization tau: {args.conformalization_tau}, seed: {seed}")

    if calibration_ratio > 0:
        x_cal, y_cal = data['x_cal'], data['y_cal']

    transformation = ConditionalIdentityTransform()
    dim_y = y_train.shape[1]
    y_grid_size = y_grid_size_per_y_dim[dim_y]
    model = NaiveMultivariateQuantileModel(input_size=x_dim, nn_input_size=x_dim + 1, output_size=dim_y, y_size=dim_y,
                                           hidden_dimensions=args.hs, dropout=args.dropout,
                                           lr=args.lr, wd=args.wd, num_ens=args.num_ens, device=args.device,
                                           y_grid_size=y_grid_size)

    loader = DataLoader(TensorDataset(x_train, y_train),
                        shuffle=True,
                        batch_size=args.bs)

    # Loss function
    loss_fn = naive_multivariate_qr_loss
    batch_loss = True
    args.tau_list = torch.Tensor([args.tau]).to(device)
    alpha = args.tau
    assert len(args.tau_list) == 1
    va_loss_list = []
    tr_loss_list = []

    for ep in tqdm(range(args.num_ep)):

        if model.done_training:
            break

        # Take train step
        ep_train_loss = []  # list of losses from each batch, for one epoch
        for (xi, yi) in loader:
            if TRAINING_OVER_ALL_QUANTILES:
                q_list = torch.rand(args.num_q)
            else:
                q_list = torch.Tensor([alpha / 2])
            loss = model.loss(loss_fn, xi, yi, q_list,
                              batch_q=batch_loss,
                              take_step=True, args=args)
            ep_train_loss.append(loss)

        ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
        tr_loss_list.append(ep_tr_loss)

        # Validation loss
        if TRAINING_OVER_ALL_QUANTILES:
            va_te_q_list = torch.linspace(0.01, 0.99, 99)
        else:
            va_te_q_list = torch.Tensor([alpha / 2, 1 - alpha / 2])
        ep_va_loss = model.update_va_loss(
            loss_fn, x_val, y_val, va_te_q_list,
            batch_q=batch_loss, curr_ep=ep, num_wait=args.wait,
            args=args
        )
        va_loss_list.append(ep_va_loss)

    params = {'dataset_name': dataset_name, 'transformation': transformation, 'epoch': model.best_va_ep[0],
              'is_real': is_real, 'seed': seed, 'tau': args.conformalization_tau,
              'vae_loss': args.vae_loss, 'vae_z_dim': args.vae_z_dim,
              'dropout': args.dropout, 'hs': str(args.hs), 'vae_mode': None, 'method_name': 'naive'}

    base_save_dir = get_save_final_figure_results_dir(**params)
    base_results_save_dir = get_save_final_results_dir(**params)
    summary_base_save_dir = get_model_summary_save_dir(**params)

    # results = evaluate_performance(model, dataset_name, x_train, y_train, x_test, y_te, is_real, scale_x, scale_y, base_save_dir,
    #             base_results_save_dir, summary_base_save_dir, is_conformalized=False, args=args)

    if args.calibration_ratio > 0:
        model.conformalize(x_cal, y_cal, args.conformalization_tau, args.tau)

        results = evaluate_performance(model, dataset_name, x_train, y_train, x_test, y_te, is_real, scale_x, scale_y,
                                       base_save_dir,
                                       base_results_save_dir, summary_base_save_dir, is_conformalized=True, args=args)
