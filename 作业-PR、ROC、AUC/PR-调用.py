import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

# -------------------------- 1. 准备数据（沿用你的原始数据） --------------------------
pp = [['T',0.9],['F',0.4],['F',0.2],['T',0.6],
      ['F',0.5],['F',0.5],['T',0.7],['T',0.4]]

# 转换为sklearn要求的数值格式：T=1（正例），F=0（负例）
y_true = np.array([1 if p[0] == 'T' else 0 for p in pp])
y_score = np.array([p[1] for p in pp])

# -------------------------- 2. 调用sklearn计算PR曲线指标 --------------------------
# 计算精确率(precision)、召回率(recall)、阈值
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
# 计算PR AUC（PR曲线下面积，衡量PR曲线性能）
pr_auc = auc(recall, precision)

# -------------------------- 3. 绘制PR曲线 --------------------------
# 解决PyCharm后端兼容问题
plt.switch_backend('TkAgg')
plt.close('all')  # 清理旧图表

# 创建画布
plt.figure(figsize=(6, 6))
# 绘制PR曲线（核心）
plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
# 绘制「随机猜测」基准线（正例占比）
pos_ratio = np.sum(y_true) / len(y_true)  # 正例在数据集中的比例
plt.plot([0, 1], [pos_ratio, pos_ratio], color='red', lw=2, linestyle='--',
         label=f'Random Guess (pos ratio = {pos_ratio:.2f})')

# 设置坐标轴范围和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (TPR)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall (PR) Curve', fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(alpha=0.3)  # 网格辅助阅读
plt.show()

# 打印关键结果
print("=== PR曲线关键指标 ===")
print(f"PR AUC值: {pr_auc:.4f}")
print(f"召回率(Recall): {np.round(recall, 4)}")
print(f"精确率(Precision): {np.round(precision, 4)}")
print(f"正例占比（随机基准）: {pos_ratio:.4f}")