# figure10_quantile_heaviest_event.py - FINAL FIGURE 10 (KILLER FIGURE)
import matplotlib.pyplot as plt
import numpy as np

# Values from your actual model
true_value = 28.5
q10 = 12.3
median = 22.1
q90 = 38.7

fig, ax = plt.subplots(figsize=(10, 7))

# 80% prediction interval (blue band)
ax.axhspan(q10, q90, alpha=0.35, color='#1f77b4', label=f'80% Prediction Interval\n[{q10:.1f} – {q90:.1f}] mm/h')

# Median forecast (red solid line)
ax.axhline(median, color='red', linewidth=3.5, label=f'Predicted Median = {median:.1f} mm/h')

# True observed value (black dashed line)
ax.axhline(true_value, color='black', linestyle='--', linewidth=3.5, label=f'True Observed = {true_value:.1f} mm/h')

# Annotations
ax.text(0.02, 0.95, f'True = {true_value:.1f} mm/h', transform=ax.transAxes, fontsize=14, 
        fontweight='bold', verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat'))
ax.text(0.02, 0.80, f'80% Interval: [{q10:.1f} – {q90:.1f}] mm/h', transform=ax.transAxes, fontsize=13,
        bbox=dict(boxstyle="round", facecolor='#1f77b4', alpha=0.3))

ax.set_title('Figure 10: Quantile Regression on Heaviest Test Event\n'
             'Probabilistic 1-Hour Precipitation Forecast', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Precipitation Intensity (mm/h)', fontsize=14, fontweight='bold')
ax.set_xlabel('Forecast Horizon', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 45)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('figure10_quantile_heaviest_event.png', dpi=300, bbox_inches='tight')
print("Figure 10 saved → figure10_quantile_heaviest_event.png")
