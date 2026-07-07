# core/experimental/

Ce dossier contient des outils **non exposés dans l'UI principale**.
Ils sont gardés en réserve pour des cas d'usage spécifiques.

---

## c4d_render_36_views.py

**Rôle :** Générer 150+ rendus multi-vues depuis un mesh Cinema 4D.

**Cas d'usage :**
- Objets complexes nécessitant une reconstruction complète à 360°
- Scènes avec plusieurs objets où TripoSplat ne suffit pas
- Qualité maximale via pipeline Splatfacto (Kaggle)

**Nécessite :**
- Cinema 4D ouvert avec le mesh chargé
- Kaggle pour l'étape Splatfacto (30h GPU/semaine gratuit)

**Pipeline :**
```
Mesh C4D → 150 rendus PNG → Kaggle Splatfacto → PLY haute qualité
```

---

## kaggle_splatfacto_pipeline.py

**Rôle :** Training 3DGS depuis rendus multi-vues via Nerfstudio/Splatfacto.

**Cas d'usage :**
- Reconstruction depuis une vidéo réelle (téléphone, caméra)
- Scènes extérieures ou intérieures avec photos multiples
- Qualité maximale quand TripoSplat n'est pas suffisant

**Nécessite :**
- Compte Kaggle (gratuit)
- 30h GPU T4/semaine incluses
- Notebook `GaussianEditor_C4D_Training.ipynb` (fourni dans le projet)

---

## Réactiver un outil

Pour réactiver un outil dormant, il suffit de l'importer depuis `core/experimental/`
et d'ajouter l'endpoint correspondant dans `core/server.py`.

Aucune modification de l'architecture principale n'est nécessaire.