import numpy
import torch


def to_tensor(data):
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, numpy.ndarray):
        if data.dtype == bool:
            return torch.from_numpy(data).bool()
        else:
            return torch.from_numpy(data).float()
    elif isinstance(data, numpy.number):
        return torch.tensor(data).float()
    else:
        print(type(data))
        raise NotImplementedError


def to_numpy(data):
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            return data.detach().cpu().numpy()
        else:
            return data.cpu().numpy()
    else:
        raise NotImplementedError


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise NotImplementedError
