import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from helper import set_seeds
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from helper import create_folder_if_it_doesnt_exist
import argparse
import os
import warnings
from datasets import datasets
from copy import deepcopy
import pickle
from plot_helper import plot_samples, render_mpl_table, float_to_str, get_conditional_data
from directories_names import get_vae_model_save_path, \
    get_cvae_model_save_name, get_encoder_base_save_dir, \
    get_encoder_transformed_y_fig_name, get_original_and_reconstructed_y_fig_name, get_encoder_summary_save_dir, \
    get_vae_results_save_dir, get_vae_results_save_path
from CVAE.models import CVAE_GAN
import math
import matplotlib
from sys import platform

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(device)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--dataset_name', type=str, default='cond_banana_k_dim_10',
                        help='dataset to use')

    parser.add_argument('--z_dim', type=int, default=3,
                        help='dimension of the embedded space')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=1000,
                        help='number of epochs')

    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--ds_type', type=str, default="SYN",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of test set size')

    parser.add_argument('--calibration_ratio', type=float, default=0.4,  # 0.5 of training size
                        help='ratio of calibration set size')

    parser.add_argument('--loss', type=str, default='KL',
                        help="use 'MMD' for MMD loss, or 'KL' for KL-divergence loss")

    parser.add_argument('--mode', type=str, default='CVAE-GAN',
                        help="one of: CVAE | CVAE-GAN | CVAE-GAN-CLASS | Bicycle | Bicycle-CLASS")

    parser.add_argument('--kl_mult', type=float, default=0.01,
                        help='KL-divergence loss multiplier')
    parser.add_argument('--suppress_plots', type=int, default=0,
                        help='1 to disable all plots, or 0 to allow plots')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    args.device = device
    args.num_ens = 1
    args.boot = 0
    args.suppress_plots = False if args.suppress_plots == 0 else 1

    if args.mode not in ['CVAE', 'CVAE-GAN', 'CVAE-GAN-CLASS', 'Bicycle', 'Bicycle-CLASS', 'CVAE-ADV']:
        print("please enter valid mode")
        assert False

    return args


def plot_y_and_reconstructed(y, y_reconstructed, base_save_dir, args, title_begin=""):
    save_path = f"{base_save_dir}/{get_original_and_reconstructed_y_fig_name()}"
    title_begin = ''
    if 'syn' in args.ds_type.lower() and y.shape[1] == 3:
        axis_lim = {'x_lim': [-2.5, 2.5], 'y_lim': [0.25, 2.2], 'z_lim': [-1.2, 1.2]}
    elif 'syn' in args.ds_type.lower() and y.shape[1] == 2:
        axis_lim = {'x_lim': [-2.5, 2.5], 'y_lim': [0.25, 2.2]}
    else:
        axis_lim = {}

    plot_samples(y, "$y$", title=title_begin + "$Y|X$",
                 axis_name='Y', a_alpha=1., save_fig_path=f'{base_save_dir}/original_y.png', args=args, **axis_lim)

    plot_samples(y, "$y$", b=y_reconstructed, b_label="$y$ reconstructed", title=title_begin + "$Y|X$",
                 axis_name='Y', a_alpha=1., b_alpha=0.5, save_fig_path=save_path, args=args, **axis_lim)


def plot_z_and_normal(z, base_save_dir, args, normal=None, title_begin=""):
    save_path = f"{base_save_dir}/{get_encoder_transformed_y_fig_name()}"
    title_begin = ''

    if normal is not None:
        plot_samples(normal, "Normal(0,1)", b=z, b_label="z",
                     title=title_begin + "Z and normal distribution", axis_name='Z', save_fig_path=save_path, args=args)
    else:
        plot_samples(z, "z", title=title_begin + "$Z|X$", axis_name='Z', save_fig_path=save_path, args=args)


def plot_transformed_and_reconstructed(vae, y_test, base_save_dir, args, x_test=None, title_begin=""):
    is_conditional = x_test is not None
    if is_conditional:
        y_rec, z = vae.encode_and_reconstruct(y_test, x_test)
    else:
        y_rec, z = vae.encode_and_reconstruct(y_test)

    plot_y_and_reconstructed(y_test.detach().cpu().numpy(), y_rec.detach().cpu().numpy(),
                             title_begin=title_begin, base_save_dir=base_save_dir, args=args)

    plot_z_and_normal(z.detach().cpu().numpy(), title_begin=title_begin,
                      base_save_dir=base_save_dir,
                      args=args)  # , normal.numpy())#, decomposed=dec_te.detach().cpu().numpy())


def train_vae_model(vae, y_train, y_val, z_dim, args, plot_losses=False, is_cond=False, x_train=None, x_val=None):
    loader = DataLoader(TensorDataset(x_train, y_train),
                        shuffle=True,
                        batch_size=args.bs)

    vae = best_vae_model = deepcopy(vae)

    val_losses = []
    train_losses = []
    KL_losses = []
    rec_losses = []
    disc_losses = []
    class_losses = []
    best_va_loss = None
    best_epoch = 0

    def perform_epoch_aux(vae, loader, n, take_vae_grad_step, take_adversary_grad_step):
        nonlocal train_losses
        for i in range(n):
            ep_train_loss = []  # list of losses from each batch, for one epoch
            for batch in loader:
                (xi, yi) = batch
                res = vae.loss(yi, xi, take_vae_grad_step, take_adversary_grad_step, take_vae_grad_step,
                               take_adversary_grad_step)
                loss = res[0]
                ep_train_loss.append(loss.cpu().item())

            ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0).item()
            train_losses += [ep_tr_loss]

    is_vanilla = vae.mode.upper() == 'CVAE' or vae.mode.upper() == 'VAE'

    if is_vanilla:
        iters_before_switch = 1

        def perform_epoch(vae, loader):
            perform_epoch_aux(vae, loader, iters_before_switch, take_vae_grad_step=True, take_adversary_grad_step=False)

    else:
        iters_before_switch = 2

        def perform_epoch(vae, loader):
            perform_epoch_aux(vae, loader, iters_before_switch, take_vae_grad_step=True, take_adversary_grad_step=False)
            perform_epoch_aux(vae, loader, iters_before_switch, take_vae_grad_step=False, take_adversary_grad_step=True)

    wait = args.wait // iters_before_switch

    for ep in tqdm(range(args.num_ep)):
        perform_epoch(vae, loader)

        # Validation loss
        with torch.no_grad():
            ep_va_loss, data_loss, kl_loss, discriminator_loss, classifier_loss = vae.loss(y_val, x_val)
            ep_va_loss, data_loss, kl_loss, \
            discriminator_loss, classifier_loss = ep_va_loss.item(), data_loss.item(), kl_loss.item(), \
                                                  discriminator_loss.item(), classifier_loss.item()
            assert not math.isnan(ep_va_loss)

        val_losses += [ep_va_loss]
        KL_losses += [kl_loss]
        rec_losses += [data_loss]
        disc_losses += [discriminator_loss]
        class_losses += [classifier_loss]
        if best_va_loss is None or ep_va_loss < best_va_loss:
            best_epoch = ep
            best_va_loss = ep_va_loss
            best_vae_model = deepcopy(vae)

        else:
            if ep - best_epoch > wait:
                break

    if plot_losses and not args.suppress_plots:
        plt.plot(val_losses)
        plt.title("val losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        plt.semilogy(KL_losses)
        plt.title("KL losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        plt.semilogy(rec_losses)
        plt.title("Reconstruction losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        if np.var(disc_losses) > 0:
            plt.title("discriminator losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.semilogy(disc_losses)
            plt.show()

        if np.var(class_losses) > 0:
            plt.title("class losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(class_losses)
            plt.show()

    return best_vae_model


def get_test_results(vae, Y, X, title_begin, save_dir):
    _, rec_loss, transformed_space_loss = vae.loss(Y, X)[:3]
    create_folder_if_it_doesnt_exist(save_dir)

    if not args.suppress_plots:
        plot_transformed_and_reconstructed(vae, y_test=Y, x_test=X, title_begin=title_begin, base_save_dir=save_dir,
                                           args=args)

    return rec_loss, transformed_space_loss


def get_losses_df(vae, dataset_name, x_test, y_te, x_train, base_save_dir, is_real, scale_x, scale_y):
    losses = {'x': [], 'rec loss': [], f'{args.loss} loss': []}

    for x_id, X, Y, x_title, _ in get_conditional_data(dataset_name, x_train, x_test, y_te, is_real, scale_x, scale_y):
        save_dir = f'{base_save_dir}/{x_title}'
        title_begin = f"{x_title}: "
        rec_loss, transformed_space_loss = get_test_results(vae, Y, X, title_begin, save_dir)
        losses['x'] += [x_id]
        losses['rec loss'] += [np.round(rec_loss.cpu().detach().numpy(), 3)]
        losses[f'{args.loss} loss'] += [np.round(transformed_space_loss.cpu().detach().numpy(), 3)]

    return losses


if __name__ == '__main__':

    args = parse_args()

    seed = args.seed
    set_seeds(seed)

    dataset_name = args.dataset_name
    print("dataset_name: ", dataset_name, "seed: ", seed)

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
    dim_y = y_train.shape[1]

    assert args.loss == 'KL'
    vae = CVAE_GAN(y_dim=dim_y, x_dim=x_dim, z_dim=args.z_dim, device=device,
                   train_size=y_train.shape[0], dropout=0.1, lr=args.lr, wd=args.wd, mode=args.mode, batch_norm=False,
                   kl_mult=args.kl_mult)

    vae = train_vae_model(vae, y_train, y_val, z_dim=args.z_dim, args=args, plot_losses=True, is_cond=True,
                          x_train=x_train, x_val=x_val)
    if vae.batch_norm:
        vae.eval()

    dataset_name = args.dataset_name
    epoch = args.num_ep
    base_save_dir = get_encoder_base_save_dir(dataset_name=dataset_name, epoch=epoch, is_real=is_real,
                                              seed=args.seed, z_dim=args.z_dim, loss=args.loss, mode=args.mode,
                                              kl_mult=args.kl_mult)
    create_folder_if_it_doesnt_exist(base_save_dir)

    models_save_path = get_vae_model_save_path(dataset_name, seed, args.loss, args.z_dim, vae_mode=args.mode)
    create_folder_if_it_doesnt_exist(models_save_path)
    file_name = get_cvae_model_save_name(dataset_name, seed, args.loss, args.z_dim, vae_mode=args.mode)

    with open(f'{file_name}', 'wb') as handle:
        pickle.dump(vae.cpu(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(file_name, 'rb') as handle:
    #     vae = pickle.load(handle).to(device)
    # if vae.batch_norm:
    #     vae.eval()

    vae = vae.to(device)

    with torch.no_grad():
        losses = get_losses_df(vae, dataset_name, x_test, y_te, x_train, base_save_dir, is_real, scale_x, scale_y)

        rec_loss, transformed_space_loss = get_test_results(vae, y_te, x_test, title_begin="marginal (test)",
                                                            save_dir=base_save_dir)
        losses['x'] += ['marginal (test)']
        losses['rec loss'] += [np.round(rec_loss.cpu().detach().numpy(), 3)]
        losses[f'{args.loss} loss'] += [np.round(transformed_space_loss.cpu().detach().numpy(), 3)]

    for key in losses:
        for i in range(len(losses[key])):
            losses[key][i] = float_to_str(losses[key][i])

    res_dict = {}
    losses = pd.DataFrame(losses)
    for i in range(len(losses.index)):
        x = losses.iloc[i]['x']
        res_dict[f'x={x}_mse_loss'] = losses.iloc[i]['rec loss']
        res_dict[f'x={x}_{args.loss}_loss'] = losses.loc[i][f'{args.loss} loss']
    res_dict = pd.DataFrame(res_dict, index=[args.seed])

    save_dir = get_vae_results_save_dir(dataset_name, epoch, is_real, seed, args.z_dim, args.loss, args.mode,
                                        kl_mult=args.kl_mult)
    create_folder_if_it_doesnt_exist(save_dir)
    save_path = get_vae_results_save_path(dataset_name, epoch, is_real, seed, args.z_dim, args.loss, args.mode,
                                          kl_mult=args.kl_mult)
    res_dict.to_csv(save_path)

    if not args.suppress_plots:
        ax = render_mpl_table(losses, header_columns=0, col_width=2.0)
        ax.set_title(f"mode = {args.mode}, loss = {args.loss}, z_dim = {args.z_dim}")
        save_dir = get_encoder_summary_save_dir(dataset_name=dataset_name, epoch=epoch, is_real=is_real,
                                                seed=args.seed, z_dim=args.z_dim, loss=args.loss, mode=args.mode,
                                                kl_mult=args.kl_mult)
        create_folder_if_it_doesnt_exist(save_dir)
        plt.savefig(f"{save_dir}/losses.png", dpi=300, bbox_inches='tight')
        plt.show()
