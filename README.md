# tsa-first-usecase

Travaux pratiques – Traitement du Signal Audio  
Mastère Data & IA – Nexa Business School

## Contenu

| Dossier | Description |
|---------|-------------|
| `data/` | Fichiers audio utilisés dans les TPs |
| `slides/` | Diaporama du Jour 1 (Signal Audio & Librosa) |
| `tp1/` | TP1 – Analyse d'un signal vocal avec Librosa |
| `tp1/outputs/` | Figures et fichiers audio générés par le TP1 |

## TP1 – Analyse d'un signal vocal

**Objectif :** explorer un signal audio numérique avec Librosa.

**Parties :**
1. Chargement & exploration (durée, samples, RMS, min/max)
2. Visualisation de la forme d'onde (waveform + zooms)
3. Calculs sur le signal (dynamique dB, énergie par moitié)
4. Expérimentation (chargement partiel, inversion, amplification +6 dB)

**Lancer le TP :**
```bash
pip install librosa matplotlib numpy soundfile
python tp1/tp1.py
```

## Environnement

- Python 3.x
- librosa, numpy, matplotlib, soundfile
