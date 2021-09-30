import torch
from helper import generate_directions
from losses import predict_y, calc_y_u
from tqdm import tqdm
from transformations import IdentityTransform

n_directions = 512


def is_in_halfspace_intersection(Y_ui, pred_yi):
    return (Y_ui >= pred_yi).all(dim=1).unsqueeze(1)


def is_in_region(model, x, y, tau, use_best_epoch=True,
                 is_transformed=False, transform=IdentityTransform(), verbose=True, n_directions=n_directions, **kwargs):
    if not is_transformed:
        y = transform.cond_transform(y, x)

    u_list, _, idx = generate_directions(y.shape[1], n_directions, tau, return_idx=True)

    coverages = perform_computation_on_predicted_halfspaces(model, x, y, u_list,
                                                            aggregation_per_batch=is_in_halfspace_intersection,
                                                            aggregation_per_direction=lambda x: x,
                                                            directions_per_batch=n_directions,
                                                            use_best_epoch=use_best_epoch,
                                                            is_transformed=is_transformed,
                                                            transform=IdentityTransform(), verbose=verbose)

    return coverages


def perform_computation_on_predicted_halfspaces(model, x, y, u_list, aggregation_per_batch,
                                                aggregation_per_direction, directions_per_batch, use_best_epoch=True,
                                                is_transformed=False, transform=IdentityTransform(), verbose=True):
    if not is_transformed:
        y = transform.cond_transform(y, x)

    if use_best_epoch:
        model = model.best_va_model[0]
    else:
        model = model.model[0]

    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    if x.shape[1] < 120:
        batch_size = 10000
    else:
        batch_size = 5000

    idx_range = range(0, y.shape[0], batch_size)
    if verbose:
        idx_range = tqdm(idx_range)

    directions_idx_range = range(0, len(u_list), directions_per_batch)

    results = []

    with torch.no_grad():
        for j in directions_idx_range:
            results_j = []
            curr_u_list = u_list[j: min(j + directions_per_batch, len(u_list))]
            for i in idx_range:
                if len(x) == 1:
                    xi = x
                else:
                    xi = x[i: min(i + batch_size, x.shape[0])]

                yi = y[i: min(i + batch_size, y.shape[0])]

                pred_yi = predict_y(model, xi, curr_u_list, get_unflattened=True)
                Y_ui = calc_y_u(curr_u_list, yi)

                res_i = aggregation_per_batch(Y_ui, pred_yi)
                results_j += [res_i.detach()]

            results += [aggregation_per_direction(torch.cat(results_j, dim=0))]

    results = torch.cat(results, dim=1).squeeze()
    return results
