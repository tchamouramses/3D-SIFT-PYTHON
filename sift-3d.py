"""
=============================================================================
  3D SIFT — Pipeline complète sur volumes IRM (PyTorch)
=============================================================================

CONTENU DE CE FICHIER :
  BLOC 1 — Noyau Gaussien 3D + convolution
  BLOC 2 — Espace-échelle + DoG
  BLOC 3 — Détection des extrema locaux
  BLOC 4 — Localisation précise (Taylor + Hessien)
  BLOC 5 — Descripteur d'apparence 64D
  BLOC 6 — Inversion d'intensité (multimodal T1↔T2)
  BLOC 7 — Mise en correspondance (ratio test de Lowe)
  BLOC 8 — Interface ligne de commande (CLI)

UTILISATION EN LIGNE DE COMMANDE :
─────────────────────────────────────────────────────────────────────────────
  1 image  (détection seule) :
    python sift3d_pytorch.py image1.hdr
    python sift3d_pytorch.py image1.nii.gz

  2 images (mise en correspondance) :
    python sift3d_pytorch.py image1.hdr image2.hdr
    python sift3d_pytorch.py image1.nii.gz image2.nii.gz

  Options :
    --octaves   INT    Nombre d'octaves              (défaut: 3)
    --scales    INT    Niveaux par octave             (défaut: 3)
    --contrast  FLOAT  Seuil de contraste             (défaut: 0.02)
    --edge      FLOAT  Seuil courbure bords           (défaut: 10.0)
    --ratio     FLOAT  Ratio test de Lowe             (défaut: 0.8)
    --downsample       Sous-échantillonne ×0.5        (si volume > 256³)
    --device    STR    'cpu' ou 'cuda'                (défaut: cpu)

EXEMPLES :
    python sift3d_pytorch.py brain.nii.gz
    python sift3d_pytorch.py t1.nii.gz t1_followup.nii.gz
    python sift3d_pytorch.py t1.nii.gz t1_followup.nii.gz --ratio 0.75 --downsample

MATHÉMATIQUES CLÉS :
  Gaussienne 3D : G(x,y,z,σ) = (1/(2πσ²)^(3/2)) · exp(-(x²+y²+z²)/(2σ²))
  DoG :           D(x,y,z,σ) = G(x,y,z,kσ) − G(x,y,z,σ)
  Gradient :      ∇D = (∂D/∂x, ∂D/∂y, ∂D/∂z, ∂D/∂σ)
  Hessien 3D :    H[i,j] = ∂²D/∂xᵢ∂xⱼ
  Localisation :  x̂ = −H⁻¹·∇D  (offset sous-pixel)
  Descripteur :   8 cellules spatiales × 8 bins sphériques = 64 valeurs
=============================================================================
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
#  STRUCTURES DE DONNÉES
# =============================================================================

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


@dataclass
class KeyPoint3D:
    """Point clé 3D (avant calcul du descripteur)."""
    x: float; y: float; z: float
    sigma: float; octave: int; scale_idx: int; response: float

    def __repr__(self):
        return (f"KeyPoint3D(pos=({self.x:.2f},{self.y:.2f},{self.z:.2f}), "
                f"σ={self.sigma:.3f}, response={self.response:.4f})")


@dataclass
class KeyPoint3DFull:
    """
    Point clé 3D avec descripteur 64D.
      descriptor  ↔ m_pucDesc[64]
    """
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


# =============================================================================
#  BLOC 1 — Noyau Gaussien 3D + convolution
# =============================================================================

def build_gaussian_kernel_3d(sigma: float, device: torch.device) -> torch.Tensor:
    """
    G(x,y,z) = (1/(2πσ²)^(3/2)) · exp(-(x²+y²+z²)/(2σ²))
    Taille du noyau : 2·⌈3σ⌉+1  (règle des 3-sigma).
    Retourne Tensor (1,1,k,k,k) pour F.conv3d.
    """
    radius = int(np.ceil(3 * sigma))
    size   = 2 * radius + 1
    coords = torch.arange(-radius, radius+1, dtype=torch.float32, device=device)
    gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(gx**2 + gy**2 + gz**2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).view(1, 1, size, size, size)


def gaussian_blur_3d(volume: torch.Tensor, sigma: float,
                     device: torch.device) -> torch.Tensor:
    kernel  = build_gaussian_kernel_3d(sigma, device)
    padding = kernel.shape[-1] // 2
    return F.conv3d(volume, kernel, padding=padding)


# =============================================================================
#  BLOC 2 — Espace-échelle + DoG
# =============================================================================

def build_scale_space(volume: torch.Tensor, config: SIFT3DConfig,
                      device: torch.device) -> Tuple[List, List]:
    """
    s+3 niveaux Gaussiens par octave → s+2 DoG.
    DoG_i = G_{i+1} − G_i  ≈ Laplacien of Gaussian.
    Sous-échantillonnage ×0.5 entre octaves.
    """
    gaussians, dogs = [], []
    cur = volume.clone()
    for octave in range(config.num_octaves):
        og = [gaussian_blur_3d(cur, config.sigma_min * (config.k**s), device)
              for s in range(config.num_scales + 3)]
        gaussians.append(og)
        dogs.append([og[i+1] - og[i] for i in range(len(og)-1)])
        d, h, w = cur.shape[2:]
        cur = F.interpolate(cur, size=(d//2, h//2, w//2),
                            mode='trilinear', align_corners=False)
    return gaussians, dogs


# =============================================================================
#  BLOC 3 — Détection des extrema locaux
# =============================================================================

def detect_extrema(dogs: List[List[torch.Tensor]], config: SIFT3DConfig,
                   device: torch.device) -> List[Tuple]:
    """
    Extrema dans voisinage 26 spatiaux (3×3×3) + 2 niveaux d'échelle.
    max_pool3d : GPU-friendly, sans boucle voxel explicite.
    """
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


# =============================================================================
#  BLOC 4 — Localisation précise (Taylor + Hessien)
# =============================================================================

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


def refine_keypoint(dogs, s, d, h, w, config):
    """
    D(x+δ) ≈ D(x)+∇Dᵀδ+½δᵀHδ  →  δ̂=−H⁻¹∇D
    Rejette si |D(x̂)|<contrast_thr ou trace²/det > (r+1)²/r.
    """
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


def detect_keypoints_3d(volume_np: np.ndarray,
                         config: SIFT3DConfig = None,
                         device_str: str = 'cpu') -> List[KeyPoint3D]:
    """Détecte les points clés 3D SIFT (blocs 1–4)."""
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
        ok,dr,hr,wr,sr,_ = refine_keypoint(dogs[oi],s,d,h,w,config)
        if not ok: continue
        sf = 2.**oi
        kps.append(KeyPoint3D(
            x=dr*sf, y=hr*sf, z=wr*sf,
            sigma=config.sigma_min*(2.**oi)*(config.k**sr),
            octave=oi, scale_idx=sr,
            response=dogs[oi][sr].squeeze()[d,h,w].item()))
    print(f"[SIFT3D] ✓ {len(kps)} points clés retenus")
    return kps


# =============================================================================
#  BLOC 5 — Descripteur 64D
#  Structure : 8 cellules spatiales (2×2×2) × 8 bins sphériques (4θ×2φ) = 64
# =============================================================================

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


def compute_descriptor_3d(volume: np.ndarray,
                           kp: KeyPoint3DFull) -> np.ndarray:
    """
    Descripteur 64D par histogramme d'orientations sphériques.
      θ = atan2(gh,gw) → 4 bins azimut
      φ = arccos(gd/m) → 2 bins élévation
    Normalisation : L2 → clip(0.2) → L2 → uint8
    """
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


def compute_all_descriptors(volume: np.ndarray,
                             keypoints: List[KeyPoint3D]) -> List[KeyPoint3DFull]:
    """Calcule les descripteurs 64D pour tous les points clés."""
    vol=(lambda v:(v-v.min())/(v.max()-v.min()+1e-8))(volume.astype(np.float32))
    feats=[]
    for kp in keypoints:
        f=KeyPoint3DFull.from_keypoint(kp); f.descriptor=compute_descriptor_3d(vol,f); feats.append(f)
    print(f"[SIFT3D] ✓ Descripteurs calculés pour {len(feats)} points clés")
    return feats


# =============================================================================
#  BLOC 6 — Inversion d'intensité (T1↔T2)
# =============================================================================
#  BLOC 7 — Mise en correspondance (ratio test de Lowe)
#  ||a-b||² = ||a||² + ||b||² − 2·aᵀb  (vectorisé, sans boucle paire)
# =============================================================================

def match_keypoints(feats1: List[KeyPoint3DFull],
                    feats2: List[KeyPoint3DFull],
                    lowe_ratio: float = 0.8,
                    use_lowe_test: bool = True) -> List[Match3D]:
    """
    Brute-force vectorisé. Garde les correspondances avec dist(NN1)/dist(NN2) < ratio.
    Résultat trié par distance croissante.
    """
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


# =============================================================================
#  Pipeline complète (blocs 1–7)
# =============================================================================

def full_pipeline(volume1: np.ndarray,
                  volume2: Optional[np.ndarray] = None,
                  config:  Optional[SIFT3DConfig] = None,
                  device_str: str = 'cpu',
                  lowe_ratio: float = 0.8,
                  multimodal: bool = False) -> dict:
    """Pipeline 3D SIFT de bout en bout. Retourne {'feats1','feats2','matches'}."""
    if config is None: config=SIFT3DConfig()
    print("\n"+"="*55+"\n  PIPELINE 3D SIFT COMPLÈTE\n"+"="*55)

    print("\n[1/3] Détection — Volume 1")
    kp1=detect_keypoints_3d(volume1,config,device_str)
    if volume2 is None:
        kp2,vol2=kp1,volume1
    else:
        print("\n[1/3] Détection — Volume 2")
        kp2=detect_keypoints_3d(volume2,config,device_str); vol2=volume2

    f1=compute_all_descriptors(volume1,kp1)
    f2=compute_all_descriptors(vol2,kp2)

    matches=match_keypoints(f1,f2,lowe_ratio=lowe_ratio)
    print(f"[SIFT3D] ✓ {len(matches)} correspondances / {len(f1)}×{len(f2)} paires")
    return {'feats1':f1,'feats2':f2,'matches':matches}


# =============================================================================
#  BLOC 8 — CLI (interface ligne de commande)
# =============================================================================

def _check_deps():
    import importlib
    missing=[p for p in ["nibabel","torch"]
             if not importlib.util.find_spec(p)]
    if missing:
        print(f"[ERREUR] Dépendances manquantes : {', '.join(missing)}")
        print(f"  → pip install {' '.join(missing)}"); sys.exit(1)


def _load_volume(path: str, downsample: bool) -> np.ndarray:
    """
    Charge un volume IRM depuis les formats suivants :
      .hdr / .img  — Analyze 7.5  (passer le fichier .hdr)
      .nii         — NIfTI non compressé
      .nii.gz      — NIfTI compressé

    Pour les fichiers Analyze, donner le chemin du .hdr ;
    nibabel trouve automatiquement le .img associé.
    """
    import nibabel as nib

    if not os.path.isfile(path):
        print(f"[ERREUR] Fichier introuvable : {path}"); sys.exit(1)

    ext = path.lower()

    # Analyze .hdr/.img : vérifier que le .img existe aussi
    if ext.endswith(".hdr"):
        img_path = path[:-4] + ".img"
        if not os.path.isfile(img_path):
            # Essayer en majuscules (certains systèmes)
            img_path = path[:-4] + ".IMG"
        if not os.path.isfile(img_path):
            print(f"[ERREUR] Fichier .img introuvable pour : {path}")
            print(f"  → Vérifie que '{os.path.basename(img_path)}' "
                  f"est dans le même dossier que le .hdr")
            sys.exit(1)

    try:
        img = nib.load(path)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le fichier : {path}")
        print(f"  → {e}"); sys.exit(1)

    vol  = img.get_fdata().astype(np.float32)
    zoom = img.header.get_zooms()

    # Garder seulement les 3 premières dimensions (certains volumes 4D)
    if vol.ndim == 4:
        print(f"    Volume 4D détecté — utilisation du premier volume (index 0)")
        vol = vol[..., 0]
    elif vol.ndim != 3:
        print(f"[ERREUR] Dimensions inattendues : {vol.shape} (attendu 3D)")
        sys.exit(1)

    fmt = "Analyze .hdr/.img" if ext.endswith(".hdr") else \
          "NIfTI .nii.gz"     if ext.endswith(".gz")  else "NIfTI .nii"

    print(f"  {path}  [{fmt}]")
    print(f"    Dimensions : {vol.shape}  |  Voxel : "
          f"{tuple(round(z, 2) for z in zoom[:3])} mm  |  "
          f"Min/Max : {vol.min():.1f}/{vol.max():.1f}")

    if downsample:
        vol = vol[::2, ::2, ::2]
        print(f"    Après ↓×2 : {vol.shape}")

    return vol


def _print_single(feats, label):
    print(f"\n{'='*55}\n  RÉSULTATS — {label}\n{'='*55}")
    print(f"  Points clés : {len(feats)}")
    if not feats: print("  (aucun — essaie --contrast 0.01)"); return
    sg=[f.sigma for f in feats]; rs=[abs(f.response) for f in feats]
    print(f"  σ  moy/min/max : {np.mean(sg):.3f}/{np.min(sg):.3f}/{np.max(sg):.3f}")
    print(f"  Réponse moy.   : {np.mean(rs):.4f}")
    print(f"  Octaves        : {sorted(set(f.octave for f in feats))}")
    print(f"\n--- 10 premiers points clés ---")
    for i,f in enumerate(feats[:10]):
        print(f"  [{i+1:2d}] ({f.x:6.1f},{f.y:6.1f},{f.z:6.1f})  "
              f"σ={f.sigma:.2f}  resp={f.response:+.4f}")


def _print_matches(f1,f2,matches,p1,p2):
    print(f"\n{'='*55}\n  RÉSULTATS — CORRESPONDANCE\n{'='*55}")
    print(f"  {os.path.basename(p1):<35} → {len(f1)} pts")
    print(f"  {os.path.basename(p2):<35} → {len(f2)} pts")
    print(f"  Correspondances retenues : {len(matches)}")
    if not matches: print("  (aucune — essaie --ratio 0.9 ou --contrast 0.01)"); return
    print(f"  Ratio moyen   : {np.mean([m.ratio for m in matches]):.3f}")
    print(f"  Distance moy. : {np.mean([m.distance for m in matches]):.1f}")
    print(f"\n--- Top 10 correspondances ---")
    for i,m in enumerate(matches[:10]):
        a,b=f1[m.idx1],f2[m.idx2]
        print(f"  [{i+1:2d}] ratio={m.ratio:.3f}  dist={m.distance:7.1f}")
        print(f"       Vol1 ({a.x:6.1f},{a.y:6.1f},{a.z:6.1f}) σ={a.sigma:.2f}")
        print(f"       Vol2 ({b.x:6.1f},{b.y:6.1f},{b.z:6.1f}) σ={b.sigma:.2f}")


def _cli():
    _check_deps()
    parser=argparse.ArgumentParser(
        description="Pipeline 3D SIFT sur images NIfTI",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("images",nargs="+",metavar="IMAGE.nii.gz")
    parser.add_argument("--octaves",  type=int,   default=3)
    parser.add_argument("--scales",   type=int,   default=3)
    parser.add_argument("--contrast", type=float, default=0.02)
    parser.add_argument("--edge",     type=float, default=10.0)
    parser.add_argument("--ratio",    type=float, default=0.8)
    parser.add_argument("--device",   type=str,   default="cpu")
    parser.add_argument("--downsample",action="store_true")
    args=parser.parse_args()
    if len(args.images)>2:
        print("[ERREUR] 1 ou 2 images maximum."); sys.exit(1)

    cfg=SIFT3DConfig(num_octaves=args.octaves,num_scales=args.scales,
                     sigma_min=1.0,contrast_threshold=args.contrast,
                     edge_threshold=args.edge)
    print("\n"+"="*55+f"\n  3D SIFT | octaves={cfg.num_octaves} scales={cfg.num_scales}"
          f" contrast={cfg.contrast_threshold} device={args.device}\n"+"="*55)
    print("\nChargement des images...")

    if len(args.images)==1:
        vol=_load_volume(args.images[0],args.downsample)
        feats=compute_all_descriptors(vol, detect_keypoints_3d(vol,cfg,args.device))
        _print_single(feats,os.path.basename(args.images[0]))
    else:
        v1=_load_volume(args.images[0],args.downsample)
        v2=_load_volume(args.images[1],args.downsample)
        res=full_pipeline(v1,v2,cfg,device_str=args.device,lowe_ratio=args.ratio)
        _print_matches(res['feats1'],res['feats2'],res['matches'],
                       args.images[0],args.images[1])


# =============================================================================
#  Point d'entrée
#    → avec arguments  : mode CLI (images NIfTI)
#    → sans arguments  : démonstration sur volume synthétique
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        print("Veuillez identifier les images hdr")
