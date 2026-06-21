import subprocess
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np

# Colab 한글 폰트 설치
subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-nanum'], capture_output=True)
fm.fontManager.__init__()  # 폰트 캐시 갱신

matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 수치 (실제 실험 결과) ──────────────────────────────────────────────────
models = ['ResNet-50', 'EfficientNet-B0']
colors = ['#1f77b4', '#ff7f0e']

val_acc    = [93.16, 88.88]          # Test Accuracy (%)
params     = [23.5, 4.0]             # 파라미터 수 (M)
train_time = [143.4, 91.8]           # 학습 시간 (분)
infer_time = [2.57, 1.31]            # 추론 시간 (ms, GPU T4 기준)

# ── 레이블 포맷 ──────────────────────────────────────────────────────────
val_labels    = [f'{v:.1f}%'  for v in val_acc]
param_labels  = [f'{p:.1f}M'  for p in params]
time_labels   = [f'{t:.1f}분' for t in train_time]
infer_labels  = [f'{i:.2f}ms' for i in infer_time]

x = np.arange(len(models))
bar_width = 0.5

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('ResNet-50 vs EfficientNet-B0 비교', fontsize=15, fontweight='bold', y=0.98)

def draw_bar(ax, values, labels, title, ylabel='', color_list=colors):
    bars = ax.bar(x, values, width=bar_width, color=color_list)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            label,
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    ax.set_ylim(0, max(values) * 1.18)

draw_bar(axes[0, 0], val_acc,    val_labels,   'Test Accuracy',              ylabel='Accuracy (%)')
draw_bar(axes[0, 1], params,     param_labels, '파라미터 수',                 ylabel='Parameters (M)')
draw_bar(axes[1, 0], train_time, time_labels,  '학습 시간',                   ylabel='시간 (분)')
draw_bar(axes[1, 1], infer_time, infer_labels, '추론 시간 (GPU T4, 1장)',     ylabel='ms / image')

plt.tight_layout()
plt.savefig('results/comparison_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료 → results/comparison_chart.png")
