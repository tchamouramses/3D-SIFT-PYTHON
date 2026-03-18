import argparse
import os
import sys

import numpy as np
from Models.sift3dconfig import SIFT3DConfig
import Functions.descriptors as Descriptor
import Functions.keypoints as Keypoint
import pipline as Pipline

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

"""
enregistrement des points cles dans un fichier txt pour la visualisation
"""
def _save_keypoints(feats, path_out):
    with open(path_out, 'w') as f:
        for kp in feats:
            f.write(f"{kp.x:.4f}  {kp.y:.4f}  {kp.z:.4f}  {kp.sigma:.4f}\n")
    print(f"  → {len(feats)} points clés sauvegardés dans '{path_out}'")


"""
Charge un volume IRM depuis les formats suivants :
    .hdr / .img  — Analyze 7.5  (passer le fichier .hdr)
    .nii         — NIfTI non compressé
    .nii.gz      — NIfTI compressé

Pour les fichiers Analyze, donner le chemin du .hdr ;
nibabel trouve automatiquement le .img associé.
"""
def _load_volume(path: str, downsample: bool) -> np.ndarray:
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

    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)  # normalise vers [0, 1]

    return vol


def _print_single(feats, label):
    print(f"\n{'='*55}\n  RÉSULTATS — {label}\n{'='*55}")
    print(f"  Points clés : {len(feats)}")
    if not feats: print("  (aucun — essaie --contrast 0.01)"); return
    sg=[f.sigma for f in feats]; rs=[abs(f.response) for f in feats]
    print(f"  σ  moy/min/max : {np.mean(sg):.3f}/{np.min(sg):.3f}/{np.max(sg):.3f}")
    print(f"  Réponse moy.   : {np.mean(rs):.4f}")
    print(f"  Octaves        : {sorted(set(f.octave for f in feats))}")
    # Sauvegarde
    out_path = os.path.splitext(label)[0] + "_keypoints.txt"
    _save_keypoints(feats, out_path)


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
        feats=Descriptor.compute_all(vol, Keypoint.detect_3d(vol,cfg,args.device))
        _print_single(feats,os.path.basename(args.images[0]))
    else:
        v1=_load_volume(args.images[0],args.downsample)
        v2=_load_volume(args.images[1],args.downsample)
        res=Pipline.initialize(v1,v2,cfg,device_str=args.device,lowe_ratio=args.ratio)
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
