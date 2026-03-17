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