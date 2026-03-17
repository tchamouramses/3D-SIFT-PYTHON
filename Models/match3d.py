from dataclasses import dataclass
import numpy as np

@dataclass
class Match3D:
    """Correspondance entre deux points clés."""
    idx1: int; idx2: int; dist_sqr: int; ratio: float

    @property
    def distance(self) -> float:
        return float(np.sqrt(self.dist_sqr))

    def __repr__(self):
        return (f"Match3D({self.idx1}↔{self.idx2}, "
                f"dist={self.distance:.1f}, ratio={self.ratio:.3f})")