# figure9_ablation_bar_chart.py - FINAL ABLATION BAR CHART (Figure 9)
import matplotlib.pyplot as plt
import numpy as np

# Data from your ablation study
models = [
    "Full Model\n(all novelties)",
    "No Multi-Scale\nAttention",
    "No Weighted\nLoss",
    "No Separate\nEmbeddings"
]
rmse = [0.3708, 0.4494, 0.3657, 0.6296]
colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#FF4757']

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(models, rmse, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)

# Add value labels on top
for bar, val in zip(bars, rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('RMSE (mm/h)', fontsize=14, fontweight='bold')
ax.set_title('Figure 9: Ablation Study — Impact of Each Novelty on RMSE\n'
             '(Lower is better | Full model = 0.3708)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.68)

# Highlight the winner
bars[0].set_edgecolor('gold')
bars[0].set_linewidth(4)

plt.tight_layout()
plt.savefig('figure9_ablation_rmse_bar_chart.png', dpi=300, bbox_inches='tight')
print("Figure 9 saved → figure9_ablation_rmse_bar_chart.png")
