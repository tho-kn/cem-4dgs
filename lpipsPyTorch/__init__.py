import torch

from .modules.lpips import LPIPS

# Small tweak so that model does not have to be loaded everytime
lpips_cache = {}

def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x (torch.Tensor): the input tensors to compare.
        y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    if (net_type, version) not in lpips_cache:
        lpips_cache[(net_type, version)] = LPIPS(net_type, version).to(device)
    criterion = lpips_cache[(net_type, version)]
    return criterion(x, y)