import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



pp = [['T',0.9],['F',0.4],['F',0.2],['T',0.6],
      ['F',0.5],['F',0.5],['T',0.7],['T',0.4]]


y_true = np.array([1 if p[0] == 'T' else 0 for p in pp])
y_score = np.array([p[1] for p in pp])


fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.switch_backend('TkAgg')
plt.close('all')


plt.figure(figsize=(6, 6))

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR/Recall)', fontsize=12)
plt.title('ROC Curve (Binary Classification)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)

plt.grid(alpha=0.3)
plt.show()

print("=== ROC曲线关键指标 ===")
print(f"AUC值: {roc_auc:.4f}")
print(f"FPR列表: {np.round(fpr, 4)}")
print(f"TPR列表: {np.round(tpr, 4)}")
print(f"对应阈值: {np.round(thresholds, 4)}")