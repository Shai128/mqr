"""
This is an implementation of VQR/VQR.R in python + our calibration

"""
import torch as torch
from helper import set_seeds, y_grid_size_per_y_dim
import argparse
import os
import warnings
from datasets import datasets
from transformations import  ConditionalIdentityTransform
from directories_names import get_save_final_figure_results_dir, get_model_summary_save_dir, get_save_final_results_dir
import ast
import matplotlib
from plot_helper import evaluate_conditional_performance
from sys import platform

from utils.q_model_ens import VectorQuantileRegression

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(device)


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
    args.fit_vqr_only = False if args.fit_vqr_only == 0 else 1

    args.tau_list = torch.Tensor([args.tau]).to(device)

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--tau', type=float, default=0.1,
                        help='quantile level')

    # parser.add_argument('--seed_begin', type=int, default=None,
    #                     help='random seed')

    parser.add_argument('--dataset_name', type=str, default='bio',
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

    parser.add_argument('--ds_type', type=str, default="REAL",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of test set size')
    parser.add_argument('--calibration_ratio', type=float, default=0.4,  # 0.5 of training size
                        help='ratio of calibration set size')

    parser.add_argument('--fit_vqr_only', type=int, default=0,
                        help='1 for True, 0 for False. If True, the program will only fit VQR, saving the '
                             'vqr results (beta1, beta2) without fitting the quantile region model')
    args = parser.parse_args()

    args = parse_args_utils(args)

    return args



if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset_name
    tau = args.tau
    print(f"dataset_name: {dataset_name}, tau: {args.tau}, conformalization tau: {args.conformalization_tau}, seed={args.seed}")


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

    y_grid_size = y_grid_size_per_y_dim[y_train.shape[1]]
    model = VectorQuantileRegression(tau, device, y_grid_size=y_grid_size)
    model.fit(dataset_name, is_real, seed, x_train, y_train)

    if args.fit_vqr_only:
        exit(0)

    transform = ConditionalIdentityTransform()
    params = {'dataset_name': dataset_name, 'transformation': transform, 'epoch': 0,
              'is_real': is_real, 'seed': seed, 'tau': args.conformalization_tau,
              'dropout': args.dropout, 'hs': str(args.hs),  'method_name': 'vector'}
    base_save_dir = get_save_final_figure_results_dir(**params)
    base_results_save_dir = get_save_final_results_dir(**params)
    summary_base_save_dir = get_model_summary_save_dir(**params)

    # evaluate_conditional_performance(model, x_train, y_train, y_train, x_test, y_te,
    #                                               base_save_dir, transform=transform, is_conformalized=False, args=args,
    #                                               dataset_name=dataset_name, scale_x=scale_x, scale_y=scale_y,
    #                                               cache=None,
    #                                               summary_base_save_dir=summary_base_save_dir,
    #                                               base_results_save_dir=base_results_save_dir, is_real=is_real)

    if calibration_ratio > 0:
        model.conformalize(x_cal, y_cal, y_train, y_train, transform, args.conformalization_tau,
                               args.tau)
        evaluate_conditional_performance(model, x_train, y_train, y_train, x_test, y_te,
                                         base_save_dir, transform, is_conformalized=True, args=args,
                                         dataset_name=dataset_name, scale_x=scale_x, scale_y=scale_y, cache=None,
                                         summary_base_save_dir=summary_base_save_dir,
                                         base_results_save_dir=base_results_save_dir, is_real=is_real)
