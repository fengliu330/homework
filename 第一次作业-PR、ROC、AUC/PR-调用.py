import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

pp = [['T',0.9],['F',0.4],['F',0.2],['T',0.6],
      ['F',0.5],['F',0.5],['T',0.7],['T',0.4]]

y_true = np.array([1 if p[0] == 'T' else 0 for p in pp])
y_score = np.array([p[1] for p in pp])

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
pr_auc = auc(recall, precision)


plt.switch_backend('TkAgg')
plt.close('all')


plt.figure(figsize=(6, 6))

plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
pos_ratio = np.sum(y_true) / len(y_true)
plt.plot([0, 1], [pos_ratio, pos_ratio], color='red', lw=2, linestyle='--',
         label=f'Random Guess (pos ratio = {pos_ratio:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (TPR)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall (PR) Curve', fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(alpha=0.3)  # 网格辅助阅读
plt.show()

print("=== PR曲线关键指标 ===")
print(f"PR AUC值: {pr_auc:.4f}")
print(f"召回率(Recall): {np.round(recall, 4)}")
print(f"精确率(Precision): {np.round(precision, 4)}")
print(f"正例占比（随机基准）: {pos_ratio:.4f}")