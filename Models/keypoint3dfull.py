from dataclasses import dataclass, field
from Models.keypoint3d import KeyPoint3D
import numpy as np

"""
Point clé 3D avec descripteur 64D.
    descriptor  ↔ m_pucDesc[64]
"""
@dataclass
class KeyPoint3DFull:
    x: float; y: float; z: float
    sigma: float; octave: int; scale_idx: int; response: float
    descriptor:  np.ndarray = field(
        default_factory=lambda: np.zeros(64, dtype=np.uint8))
    is_maxima: bool = False

    @classmethod
    def from_keypoint(cls, kp: KeyPoint3D) -> "KeyPoint3DFull":
        return cls(x=kp.x, y=kp.y, z=kp.z, sigma=kp.sigma,
                   octave=kp.octave, scale_idx=kp.scale_idx,
                   response=kp.response, is_maxima=(kp.response > 0))

    def dist_sqr(self, other: "KeyPoint3DFull") -> int:
        """Distance euclidienne² entre descripteurs — équivalent DistSqr() C++."""
        d = self.descriptor.astype(np.int32) - other.descriptor.astype(np.int32)
        return int(np.sum(d * d))

    def __repr__(self):
        return (f"KeyPoint3DFull(pos=({self.x:.1f},{self.y:.1f},{self.z:.1f}), "
                f"σ={self.sigma:.2f}, response={self.response:.4f})")

