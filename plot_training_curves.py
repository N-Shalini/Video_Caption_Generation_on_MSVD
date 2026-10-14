import json
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

base_hist_1 = json.load(open(RESULTS_DIR / "temporal_attention_history.json", encoding="utf-8"))
base_hist_2 = json.load(open(RESULTS_DIR / "temporal_attention_continued_history.json", encoding="utf-8"))
# Combine baseline history (epochs 1 to 44)
base_history = base_hist_1[:29] + base_hist_2 # epoch 30 continues from 29

enh_history = json.load(open(RESULTS_DIR / "spatiotemporal_vgg_history.json", encoding="utf-8"))

# Baseline data
base_epochs = [h["epoch"] for h in base_history]
base_train = [h["train_loss"] for h in base_history]
base_val = [h["val_loss"] for h in base_history]
base_best_idx = int(np.argmin(base_val)) if 'np' in globals() else base_val.index(min(base_val))
base_best_epoch = base_epochs[base_best_idx]
base_best_val = base_val[base_best_idx]

# Enhanced data
enh_epochs = [h["epoch"] for h in enh_history]
enh_train = [h["train_loss"] for h in enh_history]
enh_val = [h["val_loss"] for h in enh_history]
enh_best_idx = enh_val.index(min(enh_val))
enh_best_epoch = enh_epochs[enh_best_idx]
enh_best_val = enh_val[enh_best_idx]

plt.figure(figsize=(12, 5), dpi=300)

# Plot 1: Baseline Training Behavior
plt.subplot(1, 2, 1)
plt.plot(base_epochs, base_train, label="Train Loss", color="#1f77b4", linewidth=2)
plt.plot(base_epochs, base_val, label="Val Loss", color="#d62728", linewidth=2)
plt.axvline(x=base_best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best (Epoch {base_best_epoch}, {base_best_val:.3f})")
plt.scatter([base_best_epoch], [base_best_val], color="#d62728", s=60, zorder=5)
plt.title("(a) Baseline Temporal Attention Model", fontsize=12, fontweight="bold")
plt.xlabel("Training Epochs", fontsize=11)
plt.ylabel("Cross-Entropy Loss", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.ylim(2.0, 5.8)

# Plot 2: Enhanced Spatio-Temporal Model
plt.subplot(1, 2, 2)
plt.plot(enh_epochs, enh_train, label="Train Loss", color="#2ca02c", linewidth=2)
plt.plot(enh_epochs, enh_val, label="Val Loss", color="#ff7f0e", linewidth=2)
plt.axvline(x=enh_best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best (Epoch {enh_best_epoch}, {enh_best_val:.3f})")
plt.scatter([enh_best_epoch], [enh_best_val], color="#ff7f0e", s=60, zorder=5)
plt.title("(b) Enhanced Spatio-Temporal Model (Stage 2)", fontsize=12, fontweight="bold")
plt.xlabel("Training Epochs", fontsize=11)
plt.ylabel("Cross-Entropy Loss", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.ylim(2.0, 5.8)

plt.tight_layout()
output_path = RESULTS_DIR / "training_curves.png"
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved publication-quality figure to: {output_path}")
