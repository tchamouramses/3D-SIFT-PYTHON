from dataclasses import dataclass

@dataclass
class SIFT3DConfig:
    """Hyperparamètres de l'algorithme 3D SIFT."""
    num_octaves:        int   = 3
    num_scales:         int   = 4
    sigma_min:          float = 1.0
    k:                  float = 0.0   # calculé automatiquement
    contrast_threshold: float = 0.03
    edge_threshold:     float = 10.0
    max_iter_refine:    int   = 5
    offset_threshold:   float = 0.5

    def __post_init__(self):
        self.k = 2.0 ** (1.0 / self.num_scales)