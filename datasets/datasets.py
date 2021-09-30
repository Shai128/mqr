"""
Part of the code is taken from https://github.com/yromano/cqr
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import helper
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA

sys.path.insert(1, '..')


def preprocess_dataset_name(dataset_name):
    if 'k_dim' in dataset_name:
        dataset_name = dataset_name[:dataset_name.find("k_dim") - 1]
    dataset_name = dataset_name.replace("nonlinear_", "")
    return dataset_name


def plot_synthetic_dataset(k=1):
    helper.create_folder_if_it_doesnt_exist("figures")

    for x in [1.5, 2, 2.5]:
        X = (torch.Tensor([[x]]).repeat(1, k))
        Y = get_cond_dataset('nonlinear_cond_banana', 5000, X)
        Y1 = Y[:, 0]
        Y2 = Y[:, 1]
        plt.scatter(Y1, Y2, label=f'x={x}')
    plt.xlabel("Y0")
    plt.ylabel("Y1")
    plt.legend()
    plt.title("$Y \mid X = x$")
    plt.savefig("figures/conditional_syn_data.png", dpi=300, bbox_inches='tight')
    plt.show()

    Y1, Y2, _, _ = get_k_dim_banana(50000, k=k, is_nonlinear=True)
    plt.scatter(Y1, Y2)
    plt.xlabel("Y0")
    plt.ylabel("Y1")
    plt.title("$Y$")
    plt.savefig("figures/marginal_syn_data.png", dpi=300, bbox_inches='tight')
    plt.show()



def get_cond_dataset(dataset_name, n, X):
    is_nonlinear = 'nonlinear' in dataset_name
    dataset_name = preprocess_dataset_name(dataset_name)
    Y, X = get_syn_data(dataset_name, get_decomposed=False, k=X.shape[1], is_nonlinear=is_nonlinear, X=X, n=n)
    return Y


def generate_x(dataset_name, n, k):
    if dataset_name == 'cond_banana' or dataset_name == 'cond_quad_banana':
        X = torch.FloatTensor(n, k).uniform_(1 - 0.2, 3 + 0.2)
    else:
        assert False
    return X


banana_beta = None


def reset_datasets():
    global banana_beta
    banana_beta = None


def get_k_dim_banana(n, k=10, X=None, is_nonlinear=False, d=2):
    assert d in [2, 3, 4]
    global banana_beta
    if banana_beta is None:
        beta = torch.rand(k)
        beta /= beta.norm(p=1)
        banana_beta = beta

    if X is None:
        X = generate_x('cond_banana', n, k)
    else:
        assert len(X.shape) == 2
        X = X.clone().repeat(n, 1)

    X_to_output = X

    Z = (torch.rand(n) - 0.5) * 2
    one_dim_x = banana_beta @ X.T
    Z = Z * np.pi / one_dim_x

    phi = torch.rand(n) * (2 * np.pi)
    R = 0.1 * (torch.rand(n) - 0.5) * 2
    Y1 = Z + R * torch.cos(phi)
    Y2 = (-torch.cos(one_dim_x * Z) + 1) / 2 + R * torch.sin(phi)

    if is_nonlinear:
        tmp_X = X.clone()
        if len(X.shape) == 1:
            tmp_X = tmp_X.reshape(1, len(tmp_X))
        Y2 += torch.sin(tmp_X.mean(dim=1))

    decomposed = torch.cat([R.unsqueeze(-1), phi.unsqueeze(-1), Z.unsqueeze(-1)], dim=1)
    if d == 2:
        return Y1, Y2, X_to_output, decomposed
    elif d == 3:
        Y3 = torch.sin(Z)
        return Y1, Y2, Y3, X_to_output, decomposed
    elif d == 4:
        Y4 = torch.sin(Z)
        Y3 = torch.cos(torch.sin(Z)) + R * torch.sin(phi) * torch.cos(
            phi)
        return Y1, Y2, Y3, Y4, X_to_output, decomposed


def get_syn_data(dataset_name, get_decomposed=False, k=1, is_nonlinear=False, X=None, n=None):
    initial_seed = helper.get_current_seed()
    helper.set_seeds(0)
    dataset_name = preprocess_dataset_name(dataset_name)
    if n is None:
        if is_nonlinear:
            n = 20000
        elif k < 5:
            n = 20000
        elif k < 25:
            n = 20000
        elif k <= 60:
            n = 80000
        else:
            n = 100000

    if dataset_name == 'banana':
        T = (torch.rand(n) - 0.5) * 2
        phi = torch.rand(n) * (2 * np.pi)
        Z = torch.rand(n)
        R = 0.2 * Z * (1 + (1 - T.abs()) / 2)
        Y1 = T + R * torch.cos(phi)
        Y2 = T ** 2 + R * torch.sin(phi)
        decomposed = torch.cat([T.unsqueeze(-1), phi.unsqueeze(-1), R.unsqueeze(-1)], dim=1)
        Ys = [Y1, Y2]
    elif dataset_name == 'cond_banana' or dataset_name == 'sin_banana':
        Y1, Y2, X, decomposed = get_k_dim_banana(n, k=k, is_nonlinear=is_nonlinear, X=X)
        Ys = [Y1, Y2]
    elif dataset_name == 'cond_triple_banana':
        Y1, Y2, Y3, X, decomposed = get_k_dim_banana(n, k=k, is_nonlinear=is_nonlinear, d=3, X=X)
        Ys = [Y1, Y2, Y3]
    elif dataset_name == 'cond_quad_banana':
        Y1, Y2, Y3, Y4, X, decomposed = get_k_dim_banana(n, k=k, is_nonlinear=is_nonlinear, d=4, X=X)
        Ys = [Y1, Y2, Y3, Y4]
    else:
        assert False

    Y = torch.cat([y.reshape(len(y), 1) for y in Ys], dim=1)

    helper.set_seeds(initial_seed)

    if get_decomposed:
        return Y, X, decomposed

    return Y, X


def data_train_test_split(Y, X=None, device='cpu', test_ratio=0.2, val_ratio=0.2,
                          calibration_ratio=0., seed=0, scale=False, dim_to_reduce=None, is_real=True):
    data = {}
    is_conditional = X is not None
    if X is not None:
        X = X.cpu()
    Y = Y.cpu()

    y_names = ['y_train', 'y_val', 'y_te']
    if is_conditional:
        x_names = ['x_train', 'x_val', 'x_test']

        x_train, x_test, y_train, y_te = train_test_split(X, Y, test_size=test_ratio, random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, random_state=seed)

        if calibration_ratio > 0:
            x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=calibration_ratio,
                                                              random_state=seed)
            data['x_cal'] = x_cal
            data['y_cal'] = y_cal
            x_names += ['x_cal']
            y_names += ['y_cal']

        data['x_train'] = x_train
        data['x_val'] = x_val
        data['x_test'] = x_test

        if scale:
            s_tr_x = StandardScaler().fit(x_train)
            data['s_tr_x'] = s_tr_x
            for x in x_names:
                data[x] = torch.Tensor(s_tr_x.transform(data[x]))

        if (is_real and x_train.shape[1] > 70) or (dim_to_reduce is not None and x_train.shape[1] > dim_to_reduce):
            if dim_to_reduce is None:
                n_components = 50 if x_train.shape[1] < 150 else 100
            else:
                n_components = dim_to_reduce
            pca = PCA(n_components=n_components)
            pca.fit(data['x_train'])
            for x in x_names:
                data[x] = torch.Tensor(pca.transform(data[x].numpy()))

        for x in x_names:
            data[x] = data[x].to(device)

    else:
        y_train, y_te = train_test_split(Y, test_size=test_ratio, random_state=seed)
        y_train, y_val = train_test_split(y_train, test_size=val_ratio, random_state=seed)
        if calibration_ratio > 0:
            y_train, y_cal = train_test_split(y_train, test_size=calibration_ratio, random_state=seed)
            y_names += ['y_cal']
            data['y_cal'] = y_cal

    data['y_train'] = y_train
    data['y_val'] = y_val
    data['y_te'] = y_te

    if scale:
        s_tr_y = StandardScaler().fit(y_train)
        data['s_tr_y'] = s_tr_y

        for y in y_names:
            data[y] = torch.Tensor(s_tr_y.transform(data[y]))

    for y in y_names:
        data[y] = data[y].to(device)

    return data


def GetDataset(name, base_path):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

	"""

    if name == "star":
        df = pd.read_csv(base_path + 'STAR.csv')
        df.loc[df['gender'] == 'female', 'gender'] = 0
        df.loc[df['gender'] == 'male', 'gender'] = 1

        df.loc[df['ethnicity'] == 'cauc', 'ethnicity'] = 0
        df.loc[df['ethnicity'] == 'afam', 'ethnicity'] = 1
        df.loc[df['ethnicity'] == 'asian', 'ethnicity'] = 2
        df.loc[df['ethnicity'] == 'hispanic', 'ethnicity'] = 3
        df.loc[df['ethnicity'] == 'amindian', 'ethnicity'] = 4
        df.loc[df['ethnicity'] == 'other', 'ethnicity'] = 5

        df.loc[df['stark'] == 'regular', 'stark'] = 0
        df.loc[df['stark'] == 'small', 'stark'] = 1
        df.loc[df['stark'] == 'regular+aide', 'stark'] = 2

        df.loc[df['star1'] == 'regular', 'star1'] = 0
        df.loc[df['star1'] == 'small', 'star1'] = 1
        df.loc[df['star1'] == 'regular+aide', 'star1'] = 2

        df.loc[df['star2'] == 'regular', 'star2'] = 0
        df.loc[df['star2'] == 'small', 'star2'] = 1
        df.loc[df['star2'] == 'regular+aide', 'star2'] = 2

        df.loc[df['star3'] == 'regular', 'star3'] = 0
        df.loc[df['star3'] == 'small', 'star3'] = 1
        df.loc[df['star3'] == 'regular+aide', 'star3'] = 2

        df.loc[df['lunchk'] == 'free', 'lunchk'] = 0
        df.loc[df['lunchk'] == 'non-free', 'lunchk'] = 1

        df.loc[df['lunch1'] == 'free', 'lunch1'] = 0
        df.loc[df['lunch1'] == 'non-free', 'lunch1'] = 1

        df.loc[df['lunch2'] == 'free', 'lunch2'] = 0
        df.loc[df['lunch2'] == 'non-free', 'lunch2'] = 1

        df.loc[df['lunch3'] == 'free', 'lunch3'] = 0
        df.loc[df['lunch3'] == 'non-free', 'lunch3'] = 1

        df.loc[df['schoolk'] == 'inner-city', 'schoolk'] = 0
        df.loc[df['schoolk'] == 'suburban', 'schoolk'] = 1
        df.loc[df['schoolk'] == 'rural', 'schoolk'] = 2
        df.loc[df['schoolk'] == 'urban', 'schoolk'] = 3

        df.loc[df['school1'] == 'inner-city', 'school1'] = 0
        df.loc[df['school1'] == 'suburban', 'school1'] = 1
        df.loc[df['school1'] == 'rural', 'school1'] = 2
        df.loc[df['school1'] == 'urban', 'school1'] = 3

        df.loc[df['school2'] == 'inner-city', 'school2'] = 0
        df.loc[df['school2'] == 'suburban', 'school2'] = 1
        df.loc[df['school2'] == 'rural', 'school2'] = 2
        df.loc[df['school2'] == 'urban', 'school2'] = 3

        df.loc[df['school3'] == 'inner-city', 'school3'] = 0
        df.loc[df['school3'] == 'suburban', 'school3'] = 1
        df.loc[df['school3'] == 'rural', 'school3'] = 2
        df.loc[df['school3'] == 'urban', 'school3'] = 3

        df.loc[df['degreek'] == 'bachelor', 'degreek'] = 0
        df.loc[df['degreek'] == 'master', 'degreek'] = 1
        df.loc[df['degreek'] == 'specialist', 'degreek'] = 2
        df.loc[df['degreek'] == 'master+', 'degreek'] = 3

        df.loc[df['degree1'] == 'bachelor', 'degree1'] = 0
        df.loc[df['degree1'] == 'master', 'degree1'] = 1
        df.loc[df['degree1'] == 'specialist', 'degree1'] = 2
        df.loc[df['degree1'] == 'phd', 'degree1'] = 3

        df.loc[df['degree2'] == 'bachelor', 'degree2'] = 0
        df.loc[df['degree2'] == 'master', 'degree2'] = 1
        df.loc[df['degree2'] == 'specialist', 'degree2'] = 2
        df.loc[df['degree2'] == 'phd', 'degree2'] = 3

        df.loc[df['degree3'] == 'bachelor', 'degree3'] = 0
        df.loc[df['degree3'] == 'master', 'degree3'] = 1
        df.loc[df['degree3'] == 'specialist', 'degree3'] = 2
        df.loc[df['degree3'] == 'phd', 'degree3'] = 3

        df.loc[df['ladderk'] == 'level1', 'ladderk'] = 0
        df.loc[df['ladderk'] == 'level2', 'ladderk'] = 1
        df.loc[df['ladderk'] == 'level3', 'ladderk'] = 2
        df.loc[df['ladderk'] == 'apprentice', 'ladderk'] = 3
        df.loc[df['ladderk'] == 'probation', 'ladderk'] = 4
        df.loc[df['ladderk'] == 'pending', 'ladderk'] = 5
        df.loc[df['ladderk'] == 'notladder', 'ladderk'] = 6

        df.loc[df['ladder1'] == 'level1', 'ladder1'] = 0
        df.loc[df['ladder1'] == 'level2', 'ladder1'] = 1
        df.loc[df['ladder1'] == 'level3', 'ladder1'] = 2
        df.loc[df['ladder1'] == 'apprentice', 'ladder1'] = 3
        df.loc[df['ladder1'] == 'probation', 'ladder1'] = 4
        df.loc[df['ladder1'] == 'noladder', 'ladder1'] = 5
        df.loc[df['ladder1'] == 'notladder', 'ladder1'] = 6

        df.loc[df['ladder2'] == 'level1', 'ladder2'] = 0
        df.loc[df['ladder2'] == 'level2', 'ladder2'] = 1
        df.loc[df['ladder2'] == 'level3', 'ladder2'] = 2
        df.loc[df['ladder2'] == 'apprentice', 'ladder2'] = 3
        df.loc[df['ladder2'] == 'probation', 'ladder2'] = 4
        df.loc[df['ladder2'] == 'noladder', 'ladder2'] = 5
        df.loc[df['ladder2'] == 'notladder', 'ladder2'] = 6

        df.loc[df['ladder3'] == 'level1', 'ladder3'] = 0
        df.loc[df['ladder3'] == 'level2', 'ladder3'] = 1
        df.loc[df['ladder3'] == 'level3', 'ladder3'] = 2
        df.loc[df['ladder3'] == 'apprentice', 'ladder3'] = 3
        df.loc[df['ladder3'] == 'probation', 'ladder3'] = 4
        df.loc[df['ladder3'] == 'noladder', 'ladder3'] = 5
        df.loc[df['ladder3'] == 'notladder', 'ladder3'] = 6

        df.loc[df['tethnicityk'] == 'cauc', 'tethnicityk'] = 0
        df.loc[df['tethnicityk'] == 'afam', 'tethnicityk'] = 1

        df.loc[df['tethnicity1'] == 'cauc', 'tethnicity1'] = 0
        df.loc[df['tethnicity1'] == 'afam', 'tethnicity1'] = 1

        df.loc[df['tethnicity2'] == 'cauc', 'tethnicity2'] = 0
        df.loc[df['tethnicity2'] == 'afam', 'tethnicity2'] = 1

        df.loc[df['tethnicity3'] == 'cauc', 'tethnicity3'] = 0
        df.loc[df['tethnicity3'] == 'afam', 'tethnicity3'] = 1
        df.loc[df['tethnicity3'] == 'asian', 'tethnicity3'] = 2

        df = df.dropna()

        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]

        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[0:8], names[17:]))
        X = df.loc[:, data_names].values
        y = grade.values

    if name == "meps_19":
        df = pd.read_csv(base_path + 'meps_19_reg_fix.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values
        # display(df[col_names])

    if name == "meps_20":
        df = pd.read_csv(base_path + 'meps_20_reg_fix.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    if name == "meps_21":
        df = pd.read_csv(base_path + 'meps_21_reg_fix.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    if name == "facebook_1":
        df = pd.read_csv(base_path + 'facebook/Features_Variant_1.csv')
        y = df.iloc[:, 0].values
        X = df.iloc[:, 0:53].values

    if name == "facebook_2":
        df = pd.read_csv(base_path + 'facebook/Features_Variant_2.csv')
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values

    if name == "bio":
        # https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(base_path + 'CASP.csv')
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

    if name == 'blog_data':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + 'blogData_train.csv', header=None)
        X = df.iloc[:, 0:280].values
        y = df.iloc[:, -1].values

    if name == "concrete":
        dataset = np.loadtxt(open(base_path + 'Concrete_Data.csv', "rb"), delimiter=",", skiprows=1)
        X = dataset[:, :-1]
        y = dataset[:, -1:]

    if name == "bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df = pd.read_csv(base_path + 'bike_train.csv')

        # # seperating season as per values. this is bcoz this will enhance features.
        season = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season], axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather = pd.get_dummies(df['weather'], prefix='weather')
        df = pd.concat([df, weather], axis=1)

        # # # now can drop weather and season.
        df.drop(['season', 'weather'], inplace=True, axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011: 0, 2012: 1})

        df.drop('datetime', axis=1, inplace=True)
        df.drop(['casual', 'registered'], axis=1, inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        print(df.columns)
        X = df.drop('count', axis=1).values
        y = df['count'].values

    if name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace=True)
        data = pd.read_csv(base_path + 'communities.data', names=attrib['attributes'])
        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)

        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        imputer = imputer.fit(data[['OtherPerCap']])
        data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    if name == 'relation':
        # https://archive.ics.uci.edu/ml/datasets/KEGG+Metabolic+Relation+Network+%28Directed%29
        df = pd.read_csv(base_path + 'Relation Network (Directed).csv', header=None)
        df.drop([0, 2, 4, 7, 8, 10, 12, 13, 15, 17, 20], axis=1, inplace=True)
        # display(df.corr())

        feature_col = 19

        y = df.loc[:, feature_col].values
        df = df.drop(feature_col, axis=1)
        X = df.values

    if name == 'credit':
        # https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        df = pd.read_excel(base_path + 'credit.xls')
        df.columns = df.iloc[0]
        df.drop(0, axis=0, inplace=True)
        df.drop('ID', axis=1, inplace=True)
        df.drop('default payment next month', axis=1, inplace=True)

        y = df['PAY_AMT6'].values
        df = df.drop('PAY_AMT6', axis=1)
        X = df.values

    if name == 'gt':
        # https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set
        df = pd.DataFrame()
        for year in range(2011, 2016):
            df = df.append(pd.read_csv(base_path + 'gt_2015.csv'))

        y = df['AT'].values
        df = df.drop('AT', axis=1)
        X = df.values

    if name == 'house':
        df = pd.read_csv(base_path + 'kc_house_data.csv')
        y = np.array(df['price'])
        X = (df.drop(['id', 'date', 'price'], axis=1)).values

    UCI_datasets = ['kin8nm', 'naval']

    if name in UCI_datasets:
        data_dir = 'UCI_Datasets/'
        data = np.loadtxt(base_path + data_dir + name + '.txt')
        X = data[:, :-1]
        y = data[:, -1]

    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    except Exception as e:
        raise Exception("invalid dataset")

    return X, y


response_col_dict = {
    'meps_19': 3,
    'meps_20': 3,
    'meps_21': 3,
    'facebook_1': 33,
    'facebook_2': 33,
    'kin8nm': 2,
    'naval': 7,
    'bio': 6,
    'blog_data': 60,
    'relation': 12,
    'credit': 18,
    'gt': 9,
    'house': 14,
}


def get_real_data(dataset_name):
    is_1d = '1d_' in dataset_name
    dataset_name = dataset_name.replace("1d_", "")
    X, y = GetDataset(dataset_name, 'datasets/real_data/')
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    if is_1d:
        Y = y.reshape(len(y), 1)
        return Y, X

    response_col = response_col_dict[dataset_name]
    Y = torch.cat([X[:, response_col].unsqueeze(1), y.unsqueeze(1)], dim=1)

    mask = np.ones(X.shape[1], dtype=bool)
    mask[response_col] = False
    X = X[:, mask]

    return Y, X


def get_dataset(dataset_name, is_real):
    if is_real:

        Y, X = get_real_data(dataset_name)

    else:
        is_nonlinear = False
        if 'nonlinear_' in dataset_name:
            dataset_name = dataset_name.replace("nonlinear_", "")
            is_nonlinear = True

        if 'cond' in dataset_name:
            x_dim = int(re.search(r'\d+', dataset_name).group())
            Y, X = get_syn_data(dataset_name, k=x_dim, is_nonlinear=is_nonlinear)
        else:
            Y, _ = get_syn_data(dataset_name)
            X = torch.ones(len(Y)).unsqueeze(1)

    return Y, X


def get_split_data(dataset_name, is_real, device, test_ratio, val_ratio, calibration_ratio, seed, scale):
    dim_to_reduce = 10 if 'reduced_' in dataset_name else None
    dataset_name = dataset_name.replace("reduced_", "")

    Y, X = get_dataset(dataset_name, is_real)

    data = data_train_test_split(Y, X=X, device=device,
                                 test_ratio=test_ratio, val_ratio=val_ratio,
                                 calibration_ratio=calibration_ratio, seed=seed, scale=scale,
                                 dim_to_reduce=dim_to_reduce,
                                 is_real=is_real)

    if scale:
        s_tr_x, s_tr_y = data['s_tr_x'], data['s_tr_y']

        def scale_x(x):
            return torch.Tensor(s_tr_x.transform(x))

        def scale_y(y):
            return torch.Tensor(s_tr_y.transform(y))

    else:
        def scale_x(x):
            return x

        def scale_y(y):
            return y

    data['scale_x'] = scale_x
    data['scale_y'] = scale_y
    return data
