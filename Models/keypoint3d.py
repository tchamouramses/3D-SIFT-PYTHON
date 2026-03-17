from dataclasses import dataclass

@dataclass
class KeyPoint3D:
    """Point clé 3D (avant calcul du descripteur)."""
    x: float; y: float; z: float
    sigma: float; octave: int; scale_idx: int; response: float

    def __repr__(self):
        return (f"KeyPoint3D(pos=({self.x:.2f},{self.y:.2f},{self.z:.2f}), "
                f"σ={self.sigma:.3f}, response={self.response:.4f})")