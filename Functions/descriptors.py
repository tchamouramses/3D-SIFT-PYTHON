import numpy as np
from Models.keypoint3dfull import KeyPoint3DFull, KeyPoint3D
from typing import List

_DESC_SIZE  = 64
_N_ORIENT   = 8
_RADIUS_SIG = 3.0


def _local_grads(volume, cx, cy, cz, sigma):
    D,H,W = volume.shape
    r  = max(int(np.ceil(_RADIUS_SIG*sigma)),2)
    d0,d1=max(1,int(cx)-r),min(D-2,int(cx)+r+1)
    h0,h1=max(1,int(cy)-r),min(H-2,int(cy)+r+1)
    w0,w1=max(1,int(cz)-r),min(W-2,int(cz)+r+1)
    if d0>=d1 or h0>=h1 or w0>=w1:
        e=np.array([]); return e,e,e,e,e,e,e,e
    dd,dh,dw=np.mgrid[d0:d1,h0:h1,w0:w1]
    dd=dd.ravel().astype(np.int32); dh=dh.ravel().astype(np.int32); dw=dw.ravel().astype(np.int32)
    gd=(volume[dd+1,dh,dw]-volume[dd-1,dh,dw])/2.
    gh=(volume[dd,dh+1,dw]-volume[dd,dh-1,dw])/2.
    gw=(volume[dd,dh,dw+1]-volume[dd,dh,dw-1])/2.
    mag=np.sqrt(gd**2+gh**2+gw**2)
    sd=0.5*r*sigma
    gauss=np.exp(-((dd-cx)**2+(dh-cy)**2+(dw-cz)**2)/(2.*sd**2))
    return gd,gh,gw,mag*gauss,mag,dd,dh,dw


"""
Descripteur 64D par histogramme d'orientations sphériques.
    θ = atan2(gh,gw) → 4 bins azimut
    φ = arccos(gd/m) → 2 bins élévation
Normalisation : L2 → clip(0.2) → L2 → uint8
"""
def compute_3d(volume: np.ndarray,
                           kp: KeyPoint3DFull) -> np.ndarray:
    gd,gh,gw,weight,mag,dd,dh,dw = _local_grads(volume,kp.x,kp.y,kp.z,kp.sigma)
    if len(gd)==0: return np.zeros(_DESC_SIZE,dtype=np.uint8)

    cell = ((dd>=kp.x).astype(np.int32)*4 +
            (dh>=kp.y).astype(np.int32)*2 +
            (dw>=kp.z).astype(np.int32))

    theta    = np.arctan2(gh,gw)
    bin_t    = np.clip(np.floor((theta+np.pi)/(np.pi/2.)),0,3).astype(np.int32)
    m_safe   = np.where(mag>1e-8,mag,1e-8)
    phi      = np.arccos(np.clip(gd/m_safe,-1.,1.))
    bin_p    = (phi>=(np.pi/2.)).astype(np.int32)
    flat_idx = cell*_N_ORIENT + bin_t*2 + bin_p

    desc = np.zeros(_DESC_SIZE,dtype=np.float32)
    np.add.at(desc,flat_idx,weight)
    n=np.linalg.norm(desc); desc = desc/n if n>1e-8 else desc
    desc=np.clip(desc,0.,0.2)
    n2=np.linalg.norm(desc); desc = desc/n2 if n2>1e-8 else desc
    return np.clip(desc*512.,0,255).astype(np.uint8)


"""Calcule les descripteurs 64D pour tous les points clés."""
def compute_all(volume: np.ndarray,
                             keypoints: List[KeyPoint3D]) -> List[KeyPoint3DFull]:
    vol=(lambda v:(v-v.min())/(v.max()-v.min()+1e-8))(volume.astype(np.float32))
    feats=[]
    for kp in keypoints:
        f=KeyPoint3DFull.from_keypoint(kp); f.descriptor=compute_3d(vol,f); feats.append(f)
    print(f"[SIFT3D] ✓ Descripteurs calculés pour {len(feats)} points clés")
    return feats
