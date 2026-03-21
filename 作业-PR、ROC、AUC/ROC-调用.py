import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# -------------------------- 1. 准备数据（沿用你的原始数据） --------------------------
# 原始标签和预测分数
pp = [['T',0.9],['F',0.4],['F',0.2],['T',0.6],
      ['F',0.5],['F',0.5],['T',0.7],['T',0.4]]

# 提取真实标签（转换为数值：T=1，F=0）和预测分数
y_true = np.array([1 if p[0] == 'T' else 0 for p in pp])
y_score = np.array([p[1] for p in pp])

# -------------------------- 2. 调用sklearn计算ROC曲线和AUC --------------------------
# 计算FPR（假阳性率）、TPR（真阳性率）、阈值
fpr, tpr, thresholds = roc_curve(y_true, y_score)
# 计算AUC值（ROC曲线下面积）
roc_auc = auc(fpr, tpr)

# -------------------------- 3. 绘制ROC曲线 --------------------------
# 解决PyCharm后端兼容问题
plt.switch_backend('TkAgg')
# 关闭旧图表，避免内存警告
plt.close('all')

# 创建画布
plt.figure(figsize=(6, 6))
# 绘制ROC曲线
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# 绘制随机猜测的基准线（对角线）
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# 设置坐标轴范围
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# 添加标签和标题
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR/Recall)', fontsize=12)
plt.title('ROC Curve (Binary Classification)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
# 显示网格
plt.grid(alpha=0.3)
# 显示图表
plt.show()

# 可选：打印关键结果
print("=== ROC曲线关键指标 ===")
print(f"AUC值: {roc_auc:.4f}")
print(f"FPR列表: {np.round(fpr, 4)}")
print(f"TPR列表: {np.round(tpr, 4)}")
print(f"对应阈值: {np.round(thresholds, 4)}")