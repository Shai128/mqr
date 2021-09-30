import torch
import torch.nn as nn
import functools


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


class BaseModel(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims=None, n_layers=3, dropout=0., bias=True, non_linearity='lrelu',
                 batch_norm=False, last_layer=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64] * n_layers

        n_layers = len(hidden_dims)

        modules = [nn.Linear(in_dim, hidden_dims[0], bias=bias)]
        if dropout > 0:
            modules += [nn.Dropout(dropout)]
        if batch_norm:
            modules += [nn.BatchNorm1d(hidden_dims[0])]
        modules += [get_non_linearity(non_linearity)()]

        for i in range(n_layers - 1):
            modules += [nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias)]
            if batch_norm:
                modules += [nn.BatchNorm1d(hidden_dims[i + 1])]
            modules += [get_non_linearity(non_linearity)()]
            if dropout > 0:
                modules += [nn.Dropout(dropout)]

        self.intermediate_layer = nn.Sequential(*modules)

        modules += [nn.Linear(hidden_dims[-1], out_dim, bias=bias)]

        if last_layer is not None:
            modules += [last_layer()]

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

    def change_requires_grad(self, new_val):
        for p in self.parameters():
            p.requires_grad = new_val

    def freeze(self):
        self.change_requires_grad(True)

    def unfreeze(self):
        self.change_requires_grad(False)


class CVAE_GAN(nn.Module):
    def __init__(self, y_dim, x_dim, z_dim, device, train_size, mode, dropout=0.1, lr=1e-3, wd=0., batch_norm=False,
                 kl_mult=0.01):

        super().__init__()
        if x_dim <= 5:
            hidden_dims = [32, 64, 128, 256, 128, 64, 32]
            classifier_hidden_dims = [32, 64, 32]
        elif x_dim <= 8:
            hidden_dims = [64, 128, 256, 128, 64]
            classifier_hidden_dims = [32, 64, 32]
        elif x_dim <= 10:
            hidden_dims = [64, 128, 256, 512, 256, 128, 64]
            classifier_hidden_dims = [32, 64, 32]
        elif x_dim <= 25:
            hidden_dims = [64, 128, 256, 256, 128, 64]
            classifier_hidden_dims = [32, 64, 128, 64]
        elif x_dim <= 60:
            hidden_dims = [128, 256, 512, 512, 256, 128]
            classifier_hidden_dims = [32, 64, 128, 128]
        else:
            hidden_dims = [128, 256, 512, 512, 256, 128]
            classifier_hidden_dims = [32, 64, 128, 128, 256]

        self.kl_mult = kl_mult
        batch_norm = batch_norm or x_dim > 60
        self.batch_norm = batch_norm
        discriminator_hidden_dims = [64, 128, 64, 32, 16]

        encoded_dim = hidden_dims[-1]
        hidden_dims = hidden_dims[:-1]
        self.train_size = train_size

        self.features_encoder = BaseModel(x_dim + y_dim, encoded_dim, hidden_dims, dropout=dropout,
                                          batch_norm=batch_norm).to(device)
        self.features_decoder = BaseModel(encoded_dim + x_dim, y_dim, hidden_dims, dropout=dropout,
                                          batch_norm=batch_norm).to(device)  # Generator
        self.discriminator = BaseModel(x_dim + y_dim, 1, discriminator_hidden_dims, dropout=dropout,
                                       last_layer=nn.Sigmoid, batch_norm=batch_norm).to(device)
        self.classifier = BaseModel(y_dim, x_dim, classifier_hidden_dims, dropout=dropout, batch_norm=batch_norm).to(
            device)

        self.mu_linear = torch.nn.Linear(encoded_dim, z_dim).to(device)
        self.sigma_linear = torch.nn.Linear(encoded_dim, z_dim).to(device)
        self.decoder = torch.nn.Linear(z_dim, encoded_dim).to(device)

        VAE_params = []
        for model in [self.features_encoder, self.features_decoder, self.mu_linear, self.sigma_linear, self.decoder]:
            VAE_params += list(model.parameters())

        self.optimizer_G = torch.optim.Adam(VAE_params, lr=lr, weight_decay=wd)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=wd)
        self.optimizer_C = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=wd)

        self.optimizers = [self.optimizer_G, self.optimizer_D, self.optimizer_C]

        self.BCE = nn.BCELoss().to(device)
        self.z_dim = z_dim
        self.mode = mode
        self.use_discriminator = 'GAN' in mode.upper() or 'bicycle' in mode.lower()
        self.use_classifier = 'CLASS' in mode.upper()
        self.use_clr_loss = 'bicycle' in mode.lower()

    def encode_and_reconstruct(self, y, x):
        z = self.encode(y, x)
        y_rec = self.decode(z, x)
        return y_rec, z

    def encode(self, y, x):
        z, _, _ = self.get_encode_parameters(y, x)
        return z

    def get_encode_parameters(self, y, x):
        #  Sample a latent vector z given an input x from the posterior q(Z|x,y).
        encoded = self.features_encoder(torch.cat([y, x], dim=1))

        mu = self.mu_linear(encoded)
        log_sigma2 = self.sigma_linear(encoded)

        std = torch.exp(0.5 * log_sigma2)
        z = mu + torch.randn_like(std) * std

        return z, mu, log_sigma2

    def sample_y(self, x):
        z = torch.randn(x.shape[0], self.z_dim).to(x.device)
        return self.decode(z, x), z

    def decode(self, z, x):
        z_decoded = self.decoder(z)
        y_rec = self.features_decoder(torch.cat([z_decoded, x], dim=1))
        return y_rec


    def discriminator_loss(self, y, x):
        device = y.device

        label_noise = 0.
        real_label = 0
        generated_y, sample_z = self.sample_y(x)
        generated_y_label = self.discriminator(torch.cat([generated_y, x], dim=1))
        real_y_label = self.discriminator(torch.cat([y, x], dim=1))

        shift = label_noise / 2
        real_labels = torch.zeros_like(real_y_label).uniform_(real_label - shift, real_label + shift).to(
            device)
        generated_labels = torch.zeros_like(generated_y_label).uniform_(1 - real_label - shift,
                                                                        1 - real_label + shift).to(
            device)
        real_data_loss = self.BCE(real_y_label, real_labels)
        generated_data_loss = self.BCE(generated_y_label, generated_labels)

        discriminator_loss = real_data_loss + generated_data_loss
        return discriminator_loss

    def generator_losses(self, y, x, p=2):
        generator_discriminator_mult = 1e-3
        generator_classifier_mult = 1e-3
        pairwise_feature_matching_mult = 1e-3
        clr_mult = 1e-3

        generator_classifier_loss = generator_discriminator_loss = clr_loss = 0
        if self.use_discriminator or self.use_classifier or self.use_clr_loss:
            generated_y, sample_z = self.sample_y(x)

        if self.use_discriminator:
            generated_y_disc = self.discriminator.intermediate_layer(torch.cat([generated_y, x], dim=1)).mean(dim=0)
            real_y_disc = self.discriminator.intermediate_layer(torch.cat([y, x], dim=1)).mean(dim=0)
            generator_discriminator_loss = (generated_y_disc - real_y_disc).norm(p=p) / len(real_y_disc)
            generator_discriminator_loss *= generator_discriminator_mult

        if self.use_classifier:
            generated_y_class = self.classifier.intermediate_layer(generated_y)
            real_y_class = self.classifier.intermediate_layer(y)
            generator_classifier_loss = (generated_y_class - real_y_class).norm(dim=1, p=p).mean() / \
                                        generated_y_class.shape[1]
            generator_classifier_loss *= generator_classifier_mult

        if self.use_clr_loss:
            z_rec = self.encode(generated_y, x)
            clr_loss = (z_rec - sample_z).norm(dim=1, p=p).mean() / sample_z.shape[1]
            clr_loss *= clr_mult

        z, mu, log_sigma2 = self.get_encode_parameters(x=x, y=y)
        y_rec = self.decode(z=z, x=x)
        kl_weight = self.kl_mult
        kl_losses = -0.5 * torch.sum(1 + log_sigma2 - mu ** 2 - log_sigma2.exp(), dim=1)
        kl_loss = torch.mean(kl_losses, dim=0)
        kl_loss = kl_loss * kl_weight

        reconstruction_losses = ((y - y_rec) ** 2).mean(dim=1)
        reconstruction_loss = reconstruction_losses.mean()

        pairwise_feature_matching_loss = 0
        if self.use_discriminator:
            reconstructed_disc = self.discriminator.intermediate_layer(torch.cat([y_rec, x], dim=1))
            real_disc = self.discriminator.intermediate_layer(torch.cat([y, x], dim=1))
            pairwise_feature_matching_loss += (reconstructed_disc - real_disc).norm(dim=1, p=p).mean() / \
                                              real_disc.shape[1]

        if self.use_classifier:
            reconstructed_class = self.classifier.intermediate_layer(y_rec)
            real_class = self.classifier.intermediate_layer(y)
            pairwise_feature_matching_loss += (reconstructed_class - real_class).norm(dim=1, p=p).mean() / \
                                              reconstructed_class.shape[1]

        pairwise_feature_matching_loss *= pairwise_feature_matching_mult

        loss = (generator_discriminator_loss + generator_classifier_loss + kl_loss +
                reconstruction_loss + pairwise_feature_matching_loss + clr_loss)

        return loss, generator_discriminator_loss, generator_classifier_loss, kl_loss, \
               reconstruction_loss, pairwise_feature_matching_loss, clr_loss

    def classifier_loss(self, y, x, p=2):
        classifier_loss = (self.classifier(y) - x).norm(dim=1, p=p).mean()
        return classifier_loss

    def loss(self, y, x, take_vae_grad_step=False, take_adversary_grad_step=False, calc_vae_loss=True,
             calc_adversary_loss=True):

        discriminator_loss = classifier_loss = torch.Tensor([0]).to(y.device)

        if calc_vae_loss or take_vae_grad_step:
            generator_loss, generator_discriminator_loss, generator_classifier_loss, kl_loss, \
            reconstruction_loss, pairwise_feature_matching_loss, clr_loss = self.generator_losses(y, x, p=1)
            if take_vae_grad_step:
                self.optimizer_G.zero_grad()
                generator_loss.backward()
                self.optimizer_G.step()

        else:
            generator_loss = generator_discriminator_loss = generator_classifier_loss = kl_loss = \
                reconstruction_loss = pairwise_feature_matching_loss = clr_loss = torch.Tensor([0]).to(y.device)

        if calc_adversary_loss or take_adversary_grad_step:

            if self.use_discriminator:
                discriminator_loss = self.discriminator_loss(y, x)
                if take_adversary_grad_step:
                    self.optimizer_D.zero_grad()
                    discriminator_loss.backward()
                    self.optimizer_D.step()
                    self.optimizer_D.zero_grad()

            if self.use_classifier:
                classifier_loss = self.classifier_loss(y, x)
                if take_adversary_grad_step:
                    self.optimizer_C.zero_grad()
                    classifier_loss.backward()
                    self.optimizer_C.step()
                    self.optimizer_C.zero_grad()

        loss = generator_loss  # + discriminator_loss + classifier_loss
        return loss, reconstruction_loss, kl_loss, discriminator_loss, classifier_loss

    def to(self, device):
        super().to(device)
        self.optimizers = [self.optimizer_G, self.optimizer_D, self.optimizer_C]
        for optimizer in self.optimizers:
            optimizer_to(optimizer, device)
        return self


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
