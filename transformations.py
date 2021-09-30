import abc
import pickle
import torch


def function_wrapper(function, y, x):
    batch_size = 50000
    if len(y) < batch_size:
        return function(y, x)

    results = []

    for i in range(0, y.shape[0], batch_size):
        if x is None:
            curr_y = y[i: min(i + batch_size, y.shape[0])]
            curr_x = None
        else:
            curr_y, curr_x = y[i: min(i + batch_size, y.shape[0])], x[i: min(i + batch_size, x.shape[0])]
        results += [function(curr_y, curr_x)]

    return torch.cat(results)


class Trainable(abc.ABC):

    @abc.abstractmethod
    def __init__(self, model_path=None, device='cpu', **kwargs):
        super().__init__()
        assert model_path is not None

        with open(model_path, 'rb') as handle:
            self.model = pickle.load(handle).to(device)


class ConditionalTransform(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass
        # super().__init__()

    @abc.abstractmethod
    def cond_transform(self, y, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_inverse_transform(self, z, x):
        raise NotImplementedError()


class Transform(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        # super().__init__()
        pass
        # super().__init__(**kwargs)

    @abc.abstractmethod
    def cond_transform(self, y, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, z):
        raise NotImplementedError()

    def cond_inverse_transform(self, z, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()


class CVAETransform(ConditionalTransform, Trainable):

    def __init__(self, cvae_path, **kwargs):
        ConditionalTransform.__init__(self, **kwargs)
        Trainable.__init__(self, model_path=cvae_path, **kwargs)
        self.cvae = self.model

    def cond_transform(self, y, x):
        assert y.shape[0] == x.shape[0]
        with torch.no_grad():
            z = function_wrapper(self.cvae.encode, y, x)  # self.cvae.encode(y, x)
            return z

    def cond_inverse_transform(self, z, x):
        assert z.shape[0] == x.shape[0]
        with torch.no_grad():
            return function_wrapper(self.cvae.decode, z, x)  # self.cvae.decode(z, x)

    def __str__(self):
        return "cvae"


class VAETransform(Transform, Trainable):

    def __init__(self, vae_path, **kwargs):
        Transform.__init__(self, **kwargs)
        Trainable.__init__(self, model_path=vae_path, **kwargs)
        self.vae = self.model

    def transform(self, y):
        with torch.no_grad():
            z = function_wrapper(self.vae.encode, y, x=None)  # z = self.vae.encode(y)
            return z

    def cond_transform(self, y, x):
        assert x is None
        return self.transform(y)

    def inverse_transform(self, z):
        with torch.no_grad():
            return function_wrapper(self.vae.decode, z, x=None)  # self.vae.decode(z)

    def cond_inverse_transform(self, z, x):
        assert x is None
        return self.inverse_transform(z)

    def __str__(self):
        return "vae"


class IdentityTransform(Transform):

    def __init__(self, ):
        super().__init__()

    def transform(self, y):
        return y

    def cond_inverse_transform(self, z, x):
        assert x is None
        return self.inverse_transform(z)

    def cond_transform(self, y, x):
        assert x is None
        return self.transform(y)

    def inverse_transform(self, z):
        return z

    def __str__(self):
        return "identity"


class ConditionalIdentityTransform(ConditionalTransform):

    def __init__(self, ):
        super().__init__()

    def cond_transform(self, y, x):
        return y

    def cond_inverse_transform(self, z, x):
        return z

    def __str__(self):
        return "cond_identity"
