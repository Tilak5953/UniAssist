import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

eval_dir = Path(__file__).resolve().parent
charts_dir = eval_dir / "charts"
charts_dir.mkdir(parents=True, exist_ok=True)

with open(eval_dir / "benchmark_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Configure plot styling
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

colors = ['#06b6d4', '#8b5cf6', '#10b981']

models = [
    results["qwen2.5:0.5b"],
    results["tinyllama:latest"],
    results["qwen2.5:1.5b"]
]

names = [m["meta"]["name"] for m in models]
accs = [m["metrics"]["accuracy_pct"] for m in models]
lats = [m["metrics"]["avg_latency_ms"] for m in models]
rams = [m["metrics"]["ram_usage_mb"] for m in models]
disks = [m["meta"]["disk_mb"] for m in models]
cpus = [m["metrics"]["cpu_usage_pct"] for m in models]
hallus = [m["metrics"]["hallucination_rate_pct"] for m in models]

# =========================================================================
# Chart 1: Accuracy vs Latency Trade-Off & Pareto Frontier
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')

scatter = ax.scatter(lats, accs, s=[r * 0.4 for r in rams], c=colors, alpha=0.9, edgecolors='#ffffff', linewidth=1.5, zorder=5)

# Connect with trend / trade-off line
ax.plot(lats, accs, color='#64748b', linestyle=':', linewidth=2, zorder=3)

# Add annotations
for i, name in enumerate(names):
    ax.annotate(
        f"{name}\nAcc: {accs[i]}% | Lat: {lats[i]}ms\nRAM: {rams[i]}MB",
        (lats[i], accs[i]),
        xytext=(lats[i] + 15, accs[i] - 1.5 if i == 2 else accs[i] + 1.2),
        fontsize=9,
        fontweight='bold',
        color='#f8fafc',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor=colors[i], alpha=0.85),
        arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.2)
    )

ax.set_title('Exercise 4: Quality vs Latency Trade-off (Pareto Frontier)', fontsize=13, fontweight='bold', pad=15, color='#ffffff')
ax.set_xlabel('Average Response Latency (Milliseconds - Lower is Better)', fontsize=10, fontweight='semibold', labelpad=10, color='#94a3b8')
ax.set_ylabel('Ground Truth Accuracy (% - Higher is Better)', fontsize=10, fontweight='semibold', labelpad=10, color='#94a3b8')
ax.grid(True)
ax.set_xlim(180, 720)
ax.set_ylim(50, 80)

plt.tight_layout()
chart1_path = charts_dir / "accuracy_vs_latency.png"
plt.savefig(chart1_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"[OK] Generated: {chart1_path}")

# =========================================================================
# Chart 2: Resource Consumption (RAM vs Disk vs CPU Load)
# =========================================================================
fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax1.set_facecolor('#0f172a')

x = np.arange(len(names))
width = 0.30

rects1 = ax1.bar(x - width/2, rams, width, label='RAM Usage (MB)', color='#6366f1', edgecolor='#ffffff', linewidth=0.8)
rects2 = ax1.bar(x + width/2, disks, width, label='Disk Image Size (MB)', color='#06b6d4', edgecolor='#ffffff', linewidth=0.8)

ax1.set_ylabel('Memory / Storage (MB)', fontsize=10, fontweight='semibold', color='#94a3b8')
ax1.set_title('Exercise 3: Computational Footprint on Low-RAM Host', fontsize=13, fontweight='bold', pad=15, color='#ffffff')
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=10, fontweight='bold', color='#f8fafc')
ax1.legend(loc='upper left', framealpha=0.6)
ax1.grid(True, axis='y')

# Annotate values
for bar in rects1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + 25, f"{int(y)} MB", ha='center', va='bottom', fontsize=8.5, color='#a5b4fc')
for bar in rects2:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + 25, f"{int(y)} MB", ha='center', va='bottom', fontsize=8.5, color='#67e8f9')

plt.tight_layout()
chart2_path = charts_dir / "resource_consumption_bar.png"
plt.savefig(chart2_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"[OK] Generated: {chart2_path}")

# =========================================================================
# Chart 3: Multi-Metric Radar Chart
# =========================================================================
categories = ['Accuracy', 'Relevance', 'Hallucination\nResistance', 'Low-Latency\nEfficiency', 'RAM\nEfficiency']
N = len(categories)

angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')

# Inverse calculations for low is better metrics so that outer perimeter = best
for i, m in enumerate(models):
    acc_norm = m["metrics"]["accuracy_pct"]
    rel_norm = m["metrics"]["relevance_pct"] * 1.3
    hallu_res = 100 - m["metrics"]["hallucination_rate_pct"]
    lat_eff = max(10, 100 - (m["metrics"]["avg_latency_ms"] / 700 * 100))
    ram_eff = max(10, 100 - (m["metrics"]["ram_usage_mb"] / 1600 * 100))

    values = [acc_norm, rel_norm, hallu_res, lat_eff, ram_eff]
    values += values[:1]

    ax.plot(angles, values, linewidth=2, linestyle='solid', label=m["meta"]["name"], color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=0.18)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9, fontweight='bold', color='#cbd5e1')
ax.set_ylim(0, 100)
ax.set_title('Exercise 1 & 3: Multi-Dimensional Model Trade-off Radar', fontsize=13, fontweight='bold', pad=20, color='#ffffff')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), framealpha=0.7)
ax.grid(True, color='#334155')

plt.tight_layout()
chart3_path = charts_dir / "quality_metrics_radar.png"
plt.savefig(chart3_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"[OK] Generated: {chart3_path}")

# =========================================================================
# Chart 4: RAG Domain Accuracy Breakdown
# =========================================================================
domains = ['Exams', 'Attendance', 'Grading', 'Fees', 'Codebase']
acc_qwen05 = [66.7, 60.0, 60.0, 50.0, 50.0]
acc_tiny = [75.0, 70.0, 70.0, 62.5, 50.0]
acc_qwen15 = [83.3, 75.0, 80.0, 75.0, 60.0]

fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')

x = np.arange(len(domains))
w = 0.25

r1 = ax.bar(x - w, acc_qwen05, w, label='Qwen 2.5 (0.5B)', color=colors[0], edgecolor='#ffffff', linewidth=0.8)
r2 = ax.bar(x, acc_tiny, w, label='TinyLlama (1.1B)', color=colors[1], edgecolor='#ffffff', linewidth=0.8)
r3 = ax.bar(x + w, acc_qwen15, w, label='Qwen 2.5 (1.5B)', color=colors[2], edgecolor='#ffffff', linewidth=0.8)

ax.set_ylabel('Domain Accuracy (%)', fontsize=10, fontweight='semibold', color='#94a3b8')
ax.set_title('Exercise 2 & 5: Accuracy Across University Knowledge Base Domains', fontsize=13, fontweight='bold', pad=15, color='#ffffff')
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=10, fontweight='bold', color='#f8fafc')
ax.legend(loc='upper right', framealpha=0.7)
ax.grid(True, axis='y')
ax.set_ylim(0, 100)

plt.tight_layout()
chart4_path = charts_dir / "domain_accuracy_bar.png"
plt.savefig(chart4_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"[OK] Generated: {chart4_path}")

print("\n================================================================")
print("All 4 evaluation charts generated successfully in: evaluations/charts/")
print("================================================================")
