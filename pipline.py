from typing import Optional
import numpy as np
from Models.sift3dconfig import SIFT3DConfig
import Functions.keypoints as Keypoint
import Functions.descriptors as Descriptor

"""Pipeline 3D SIFT de bout en bout. Retourne {'feats1','feats2','matches'}."""
def initialize(volume1: np.ndarray,
                  volume2: Optional[np.ndarray] = None,
                  config:  Optional[SIFT3DConfig] = None,
                  device_str: str = 'cpu',
                  lowe_ratio: float = 0.8,
                  multimodal: bool = False) -> dict:
    if config is None: config=SIFT3DConfig()
    print("\n"+"="*55+"\n  PIPELINE 3D SIFT COMPLÈTE\n"+"="*55)

    print("\n[1/3] Détection — Volume 1")
    kp1=Keypoint.detect_3d(volume1,config,device_str)
    if volume2 is None:
        kp2,vol2=kp1,volume1
    else:
        print("\n[1/3] Détection — Volume 2")
        kp2=Keypoint.detect_3d(volume2,config,device_str); vol2=volume2

    f1=Descriptor.compute_all(volume1,kp1)
    f2=Descriptor.compute_all(vol2,kp2)

    matches=Keypoint.match(f1,f2,lowe_ratio=lowe_ratio)
    print(f"[SIFT3D] ✓ {len(matches)} correspondances / {len(f1)}×{len(f2)} paires")
    return {'feats1':f1,'feats2':f2,'matches':matches}
