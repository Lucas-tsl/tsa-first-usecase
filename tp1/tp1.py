"""
TP1 – Analyse d'un signal vocal avec Librosa
Mastère Data & IA – Nexa Business School
Durée : 45 min | En binôme
"""

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

AUDIO_FILE = "/sessions/optimistic-tender-hypatia/mnt/uploads/NHU05104095.wav"
OUT_DIR    = "/sessions/optimistic-tender-hypatia/mnt/Detection"

# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 1 – Chargement & exploration (10 min)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PARTIE 1 – Chargement & exploration")
print("=" * 60)

# Chargement avec sr forcé à 22050
y, sr = librosa.load(AUDIO_FILE, sr=22050)

duree      = len(y) / sr
nb_samples = len(y)
amp_min    = y.min()
amp_max    = y.max()
rms        = np.sqrt(np.mean(y ** 2))

print(f"  Fichier       : {AUDIO_FILE.split('/')[-1]}")
print(f"  sr (forcé)    : {sr} Hz")
print(f"  Durée         : {duree:.3f} s")
print(f"  Nb échantillons : {nb_samples}")
print(f"  Amplitude min : {amp_min:.4f}")
print(f"  Amplitude max : {amp_max:.4f}")
print(f"  RMS           : {rms:.6f}")

# Comparaison sr=None vs sr=22050
y_orig, sr_orig = librosa.load(AUDIO_FILE, sr=None)
print(f"\n  [Comparaison sr=None]")
print(f"  sr original   : {sr_orig} Hz")
print(f"  Nb samples (orig) : {len(y_orig)}")
print(f"  Nb samples (22050): {len(y)}")
if sr_orig != 22050:
    print(f"  → Rééchantillonnage : {sr_orig} Hz → 22050 Hz")
else:
    print(f"  → sr original = 22050 Hz, aucun rééchantillonnage nécessaire")

# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 2 – Visualisation waveform (15 min)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PARTIE 2 – Visualisation waveform")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Waveform complète
librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#00B4D8')
axes[0].set_title("Forme d'onde complète", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Temps (s)")
axes[0].set_ylabel("Amplitude")
axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')

# Zoom 0 → 0.5 s
start_z = 0
end_z   = int(0.5 * sr)
t_zoom  = np.arange(end_z - start_z) / sr
axes[1].plot(t_zoom, y[start_z:end_z], color='#F77F00', linewidth=0.8)
axes[1].set_title("Zoom : 0.0 s → 0.5 s (phonèmes / syllabes)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Temps (s)")
axes[1].set_ylabel("Amplitude")
axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')

# Zoom 0.5 → 1.0 s
start_z2 = int(0.5 * sr)
end_z2   = int(1.0 * sr)
t_zoom2  = np.arange(end_z2 - start_z2) / sr + 0.5
axes[2].plot(t_zoom2, y[start_z2:end_z2], color='#06D6A0', linewidth=0.8)
axes[2].set_title("Zoom : 0.5 s → 1.0 s", fontsize=13, fontweight='bold')
axes[2].set_xlabel("Temps (s)")
axes[2].set_ylabel("Amplitude")
axes[2].axhline(0, color='gray', linewidth=0.5, linestyle='--')

plt.suptitle("TP1 – Partie 2 : Visualisation Waveform", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/tp1_partie2_waveform.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figure sauvegardée : tp1_partie2_waveform.png")

# Identification silences / zones actives (seuil RMS sur fenêtres)
frame_len   = int(0.05 * sr)  # fenêtres de 50 ms
hop         = frame_len // 2
energies    = np.array([
    np.sqrt(np.mean(y[i:i+frame_len]**2))
    for i in range(0, len(y) - frame_len, hop)
])
seuil       = rms * 0.2
silences    = np.where(energies < seuil)[0]
actives     = np.where(energies >= seuil)[0]

if len(silences) > 0:
    print(f"  Silence détecté (ex.) : ~{silences[0]*hop/sr:.2f}s – {(silences[0]*hop+frame_len)/sr:.2f}s")
    if len(silences) > 1:
        print(f"  Silence détecté (ex.) : ~{silences[-1]*hop/sr:.2f}s – {(silences[-1]*hop+frame_len)/sr:.2f}s")
if len(actives) > 0:
    print(f"  Zone active (ex.)     : ~{actives[0]*hop/sr:.2f}s – {(actives[0]*hop+frame_len)/sr:.2f}s")
    if len(actives) > 1:
        print(f"  Zone active (ex.)     : ~{actives[-1]*hop/sr:.2f}s – {(actives[-1]*hop+frame_len)/sr:.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 3 – Calculs sur le signal (10 min)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PARTIE 3 – Calculs sur le signal")
print("=" * 60)

# Dynamique en dB
peak    = np.max(np.abs(y))
dynamique_dB = 20 * np.log10(peak / (rms + 1e-10))
print(f"  Peak          : {peak:.4f}")
print(f"  RMS           : {rms:.6f}")
print(f"  Dynamique     : {dynamique_dB:.2f} dB  [20×log10(peak/RMS)]")

# Énergie totale et par moitié
mid         = len(y) // 2
energie_tot = np.sum(y ** 2)
energie_1   = np.sum(y[:mid] ** 2)
energie_2   = np.sum(y[mid:] ** 2)
pct1        = energie_1 / energie_tot * 100
pct2        = energie_2 / energie_tot * 100

print(f"\n  Énergie totale  : {energie_tot:.4f}")
print(f"  Énergie 1ère moitié : {energie_1:.4f}  ({pct1:.1f}%)")
print(f"  Énergie 2ème moitié : {energie_2:.4f}  ({pct2:.1f}%)")

if energie_1 > energie_2:
    print(f"  → La 1ère moitié est plus énergétique (+{(energie_1/energie_2 - 1)*100:.1f}%)")
else:
    print(f"  → La 2ème moitié est plus énergétique (+{(energie_2/energie_1 - 1)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 4 – Expérimentation (10 min)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PARTIE 4 – Expérimentation")
print("=" * 60)

# 4a – Charger uniquement les 3 premières secondes
y3s, sr3 = librosa.load(AUDIO_FILE, sr=22050, duration=3.0)
print(f"  Chargement 3 premières secondes :")
print(f"    Nb samples : {len(y3s)}  (attendu ≈ {3*sr})")
print(f"    Durée réelle : {len(y3s)/sr3:.3f} s")

# 4b – Signal inversé
y_inv = y[::-1].copy()
print(f"\n  Signal inversé (y[::-1]) :")
print(f"    Même shape : {y_inv.shape}  |  min={y_inv.min():.4f}  max={y_inv.max():.4f}")

# Sauvegarde du signal inversé
import soundfile as sf
sf.write(f"{OUT_DIR}/signal_inverse.wav", y_inv, sr)
print(f"    Fichier inversé sauvegardé : signal_inverse.wav")

# 4c – Amplification +6 dB (×2)
y_amp = y * 2.0
n_clipped = np.sum(np.abs(y_amp) > 1.0)
pct_clip  = n_clipped / len(y_amp) * 100
y_amp_clip = np.clip(y_amp, -1.0, 1.0)
print(f"\n  Amplification +6 dB (×2) :")
print(f"    Échantillons > 1.0 (clipping) : {n_clipped}  ({pct_clip:.2f}%)")
if n_clipped > 0:
    print(f"    → CLIPPING détecté ! {pct_clip:.1f}% du signal est écrêté → distorsion audible")
else:
    print(f"    → Pas de clipping (signal bien en deçà de la saturation)")

# Figure synthèse Partie 4
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Signal 3s
librosa.display.waveshow(y3s, sr=sr, ax=axes[0], color='#9D4EDD')
axes[0].set_title("Signal chargé sur 3 s (duration=3)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Temps (s)"); axes[0].set_ylabel("Amplitude")

# Signal inversé
librosa.display.waveshow(y_inv, sr=sr, ax=axes[1], color='#F77F00')
axes[1].set_title("Signal inversé y[::-1]", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Temps (s)"); axes[1].set_ylabel("Amplitude")

# Signal amplifié avec zones de clipping
t_ax = np.linspace(0, duree, len(y_amp))
axes[2].plot(t_ax, y_amp, color='#00B4D8', linewidth=0.5, label='Signal ×2', alpha=0.7)
axes[2].plot(t_ax, y_amp_clip, color='#EF233C', linewidth=0.5, label='Après clip', alpha=0.8)
axes[2].axhline( 1.0, color='white', linewidth=1, linestyle='--', label='Limite ±1.0')
axes[2].axhline(-1.0, color='white', linewidth=1, linestyle='--')
axes[2].set_title(f"Amplification +6 dB (×2) – Clipping sur {pct_clip:.1f}% des samples", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Temps (s)"); axes[2].set_ylabel("Amplitude")
axes[2].legend(loc='upper right', fontsize=9)

plt.suptitle("TP1 – Partie 4 : Expérimentation", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/tp1_partie4_experimentation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  Figure sauvegardée : tp1_partie4_experimentation.png")

print("\n" + "=" * 60)
print("TP1 terminé – Fichiers produits :")
print("  tp1_partie2_waveform.png")
print("  tp1_partie4_experimentation.png")
print("  signal_inverse.wav")
print("=" * 60)
