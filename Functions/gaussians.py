
import torch
import torch.nn.functional as F
import numpy as np
from Models.sift3dconfig import SIFT3DConfig
from typing import List, Tuple

"""
G(x,y,z) = (1/(2πσ²)^(3/2)) · exp(-(x²+y²+z²)/(2σ²))
Taille du noyau : 2·⌈3σ⌉+1  (règle des 3-sigma).
Retourne Tensor (1,1,k,k,k) pour F.conv3d.
"""
def build_3D_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = int(np.ceil(3 * sigma))
    size   = 2 * radius + 1
    coords = torch.arange(-radius, radius+1, dtype=torch.float32, device=device)
    gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(gx**2 + gy**2 + gz**2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).view(1, 1, size, size, size)


def blur_3d(volume: torch.Tensor, sigma: float,
                     device: torch.device) -> torch.Tensor:
    kernel  = build_3D_kernel(sigma, device)
    padding = kernel.shape[-1] // 2
    return F.conv3d(volume, kernel, padding=padding)

"""
s+3 niveaux Gaussiens par octave → s+2 DoG.
DoG_i = G_{i+1} − G_i  ≈ Laplacien of Gaussian.
Sous-échantillonnage ×0.5 entre octaves.
"""
def build_scale_space(volume: torch.Tensor, config: SIFT3DConfig,
                      device: torch.device) -> Tuple[List, List]:
    gaussians, dogs = [], []
    cur = volume.clone()
    for octave in range(config.num_octaves):
        og = [blur_3d(cur, config.sigma_min * (config.k**s), device)
              for s in range(config.num_scales + 3)]
        gaussians.append(og)
        dogs.append([og[i+1] - og[i] for i in range(len(og)-1)])
        d, h, w = cur.shape[2:]
        cur = F.interpolate(cur, size=(d//2, h//2, w//2),
                            mode='trilinear', align_corners=False)
    return gaussians, dogs