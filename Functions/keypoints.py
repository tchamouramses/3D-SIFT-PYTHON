import torch
import numpy as np
import torch.nn.functional as F
from Models.sift3dconfig import SIFT3DConfig
from typing import List, Tuple
from Models.keypoint3dfull import KeyPoint3DFull, KeyPoint3D
from Models.match3d import Match3D
from Functions.gaussians import build_scale_space


"""
D(x+δ) ≈ D(x)+∇Dᵀδ+½δᵀHδ  →  δ̂=−H⁻¹∇D
Rejette si |D(x̂)|<contrast_thr ou trace²/det > (r+1)²/r.
"""
def refine(dogs, s, d, h, w, config):
    n_dogs = len(dogs)
    sh = dogs[s].squeeze().shape
    md,mh,mw = sh[0]-2, sh[1]-2, sh[2]-2

    # Valeurs par défaut pour g et off (au cas où la boucle ne s'exécute pas)
    g   = torch.zeros(4, dtype=torch.float64)
    off = torch.zeros(4, dtype=torch.float64)

    for _ in range(config.max_iter_refine):
        # Vérification des bornes AVANT chaque accès
        if not (1 <= s <= n_dogs-2): return False,0.,0.,0.,0,0.
        sh2 = dogs[s].squeeze().shape
        if not (1 <= d <= sh2[0]-2 and
                1 <= h <= sh2[1]-2 and
                1 <= w <= sh2[2]-2):
            return False,0.,0.,0.,0,0.

        g = _grad(dogs,s,d,h,w)
        H = _hess(dogs,s,d,h,w)
        try:
            off,_,_,_ = torch.linalg.lstsq(H,-g.unsqueeze(1))
            off = off.squeeze()
        except Exception:
            return False,0.,0.,0.,0,0.

        if off.abs().max() < config.offset_threshold:
            break

        d += int(torch.round(off[0]).item())
        h += int(torch.round(off[1]).item())
        w += int(torch.round(off[2]).item())
        s += int(torch.round(off[3]).item())

    # Vérification finale des bornes après la boucle
    # (s, d, h, w ont peut-être été mis à jour à la dernière itération)
    if not (1 <= s <= n_dogs-2): return False,0.,0.,0.,0,0.
    sh3 = dogs[s].squeeze().shape
    if not (1 <= d <= sh3[0]-2 and
            1 <= h <= sh3[1]-2 and
            1 <= w <= sh3[2]-2):
        return False,0.,0.,0.,0,0.

    resp = dogs[s].squeeze()[d,h,w] + 0.5*(g@off)
    if resp.abs() < config.contrast_threshold: return False,0.,0.,0.,0,0.

    H3=_hess(dogs,s,d,h,w)[:3,:3]; tr=H3.trace(); det=torch.linalg.det(H3)
    if det<=0: return False,0.,0.,0.,0,0.
    r=config.edge_threshold
    if (tr**2/det)>(r+1)**2/r: return False,0.,0.,0.,0,0.

    return True, d+off[0].item(), h+off[1].item(), w+off[2].item(), s, off[3].item()



"""Détecte les points clés 3D SIFT (blocs 1-4)."""
def detect_3d(volume_np: np.ndarray,
                         config: SIFT3DConfig = None,
                         device_str: str = 'cpu') -> List[KeyPoint3D]:
    if config is None: config = SIFT3DConfig()
    device = torch.device(
        device_str if (device_str=='cpu' or torch.cuda.is_available()) else 'cpu')
    print(f"[SIFT3D] Volume {volume_np.shape} | device={device} | "
          f"{config.num_octaves} octaves | {config.num_scales} scales")
    vol = volume_np.astype(np.float32)
    vol = (vol-vol.min())/(vol.max()-vol.min()+1e-8)
    t   = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

    print("[SIFT3D] Étape 1/4 : Espace-échelle Gaussien...")
    gaussians, dogs = build_scale_space(t, config, device)
    print(f"[SIFT3D] Étape 2/4 : {sum(len(d) for d in dogs)} cartes DoG")
    print("[SIFT3D] Étape 3/4 : Détection des extrema...")
    cands = detect_extrema(dogs, config, device)
    print(f"[SIFT3D]  → {len(cands)} candidats")
    print("[SIFT3D] Étape 4/4 : Localisation précise...")
    kps = []
    for oi,s,d,h,w in cands:
        ok,dr,hr,wr,sr,_ = refine(dogs[oi],s,d,h,w,config)
        if not ok: continue
        sf = 2.**oi
        kps.append(KeyPoint3D(
            x=dr*sf, y=hr*sf, z=wr*sf,
            sigma=config.sigma_min*(2.**oi)*(config.k**sr),
            octave=oi, scale_idx=sr,
            response=dogs[oi][sr].squeeze()[d,h,w].item()))
    print(f"[SIFT3D] ✓ {len(kps)} points clés retenus")
    return kps


"""
Brute-force vectorisé. Garde les correspondances avec dist(NN1)/dist(NN2) < ratio.
Résultat trié par distance croissante.
"""
def match(feats1: List[KeyPoint3DFull],
                    feats2: List[KeyPoint3DFull],
                    lowe_ratio: float = 0.8,
                    use_lowe_test: bool = True) -> List[Match3D]:
    if not feats1 or not feats2: return []
    d1=np.stack([f.descriptor for f in feats1]).astype(np.int32)
    d2=np.stack([f.descriptor for f in feats2]).astype(np.int32)
    n1=np.sum(d1**2,axis=1,keepdims=True); n2=np.sum(d2**2,axis=1,keepdims=True)
    dmat=np.clip(n1+n2.T-2*(d1@d2.T),0,None)
    matches=[]
    for i,row in enumerate(dmat):
        if use_lowe_test and len(feats2)>=2:
            idx=np.argpartition(row,2)[:2]; idx=idx[np.argsort(row[idx])]
            j1,j2=idx[0],idx[1]
            r1=float(np.sqrt(max(row[j1],0))); r2=float(np.sqrt(max(row[j2],1)))
            ratio=r1/r2 if r2>1e-6 else 1.
            if ratio<lowe_ratio: matches.append(Match3D(i,int(j1),int(row[j1]),ratio))
        else:
            j=int(np.argmin(row)); matches.append(Match3D(i,j,int(row[j]),0.))
    matches.sort(key=lambda m:m.dist_sqr)
    return matches


"""
Extrema dans voisinage 26 spatiaux (3x3x3) + 2 niveaux d'échelle.
max_pool3d : GPU-friendly, sans boucle voxel explicite.
"""
def detect_extrema(dogs: List[List[torch.Tensor]], config: SIFT3DConfig,
                   device: torch.device) -> List[Tuple]:
    candidates = []
    for oi, od in enumerate(dogs):
        for s in range(1, len(od)-1):
            Dp, Dc, Dn = od[s-1], od[s], od[s+1]
            pm  = F.max_pool3d(Dc, 3, stride=1, padding=1)
            isM = (Dc == pm) & (Dc >= Dp) & (Dc >= Dn)
            nm  = -F.max_pool3d(-Dc, 3, stride=1, padding=1)
            ism = (Dc == nm) & (Dc <= Dp) & (Dc <= Dn)
            msk = (isM | ism) & (Dc.abs() > config.contrast_threshold)
            # Exclure les bords
            msk[:,:,:1,:,:]=msk[:,:,-1:,:,:]=False
            msk[:,:,:,:1,:]=msk[:,:,:,-1:,:]=False
            msk[:,:,:,:,:1]=msk[:,:,:,:,-1:]=False
            for p in msk.squeeze().nonzero(as_tuple=False):
                candidates.append((oi, s, p[0].item(), p[1].item(), p[2].item()))
    return candidates


def _grad(dogs, s, d, h, w):
    D,Dp,Dn = dogs[s].squeeze(), dogs[s-1].squeeze(), dogs[s+1].squeeze()
    return torch.tensor([
        (D[d+1,h,w]-D[d-1,h,w])/2., (D[d,h+1,w]-D[d,h-1,w])/2.,
        (D[d,h,w+1]-D[d,h,w-1])/2., (Dn[d,h,w] -Dp[d,h,w] )/2.,
    ], dtype=torch.float64)


def _hess(dogs, s, d, h, w):
    D,Dp,Dn = dogs[s].squeeze(), dogs[s-1].squeeze(), dogs[s+1].squeeze()
    v=D[d,h,w]
    ddd=D[d+1,h,w]+D[d-1,h,w]-2*v; dhh=D[d,h+1,w]+D[d,h-1,w]-2*v
    dww=D[d,h,w+1]+D[d,h,w-1]-2*v; dss=Dn[d,h,w]+Dp[d,h,w]-2*v
    ddh=(D[d+1,h+1,w]-D[d+1,h-1,w]-D[d-1,h+1,w]+D[d-1,h-1,w])/4
    ddw=(D[d+1,h,w+1]-D[d+1,h,w-1]-D[d-1,h,w+1]+D[d-1,h,w-1])/4
    dhw=(D[d,h+1,w+1]-D[d,h+1,w-1]-D[d,h-1,w+1]+D[d,h-1,w-1])/4
    dds=(Dn[d+1,h,w]-Dn[d-1,h,w]-Dp[d+1,h,w]+Dp[d-1,h,w])/4
    dhs=(Dn[d,h+1,w]-Dn[d,h-1,w]-Dp[d,h+1,w]+Dp[d,h-1,w])/4
    dws=(Dn[d,h,w+1]-Dn[d,h,w-1]-Dp[d,h,w+1]+Dp[d,h,w-1])/4
    return torch.tensor([[ddd,ddh,ddw,dds],[ddh,dhh,dhw,dhs],
                          [ddw,dhw,dww,dws],[dds,dhs,dws,dss]],
                         dtype=torch.float64)