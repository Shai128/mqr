import torch


def predict_y(model, x, u_list, get_unflattened=False):  # predict directional quantile
    num_pts = len(x)
    num_u = u_list.shape[0]

    u_rep = u_list.repeat(num_pts, 1)

    if x is None:
        model_in = u_rep
    else:
        x_stacked = x.unsqueeze(1).repeat(1, num_u, 1).flatten(0, 1)
        model_in = torch.cat([x_stacked, u_rep], dim=1)

    model_in.requires_grad = True
    pred_coeffs = model(model_in)

    pred = pred_coeffs[:, 0]
    if get_unflattened:
        unflatten = torch.nn.Unflatten(0, (num_pts, num_u))
        pred = unflatten(pred)

    return pred


def calc_y_u(u_list, y):
    num_pts = len(y)
    Y_u = torch.bmm(u_list.unsqueeze(0).repeat(num_pts, 1, 1), y.unsqueeze(-1)).squeeze(-1)

    return Y_u


def multivariate_qr_loss(model, y, x, u_list, device, args):
    tau_list = args.tau_list  # ,args.gamma
    pred = predict_y(model=model, x=x, u_list=u_list, get_unflattened=True)
    Y_u = calc_y_u(u_list, y)

    diff = Y_u - pred
    flattened_diff = diff.flatten()

    mask = (tau_list - flattened_diff.le(0).float()).detach()

    pinball_loss = ((mask * flattened_diff).mean())

    return pinball_loss


def naive_multivariate_qr_loss(model, y, x, q_list, device, args):
    num_pts = y.size(0)

    with torch.no_grad():
        l_list = torch.min(torch.stack([q_list, 1 - q_list], dim=1), dim=1)[0].to(device)
        u_list = 1.0 - l_list

    q_list = torch.cat([l_list, u_list], dim=0)
    num_q = q_list.shape[0]

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)

    if x is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)

    pred_y = model(model_in)

    diff = pred_y - y_stacked
    mask = (diff.ge(0).float() - q_rep).detach()  # / q_rep

    pinball_loss = (mask * diff).mean(dim=0).sum()

    return pinball_loss
