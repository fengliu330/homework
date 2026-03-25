import numpy as np
import matplotlib

# 解决PyCharm绘图兼容问题
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
y_true = np.array([
    [0, 0, 1],  # 样本1-类别2
    [0, 1, 0],  # 样本2-类别1
    [1, 0, 0],  # 样本3-类别0
    [0, 0, 1],  # 样本4-类别2
    [1, 0, 0],  # 样本5-类别0
    [0, 1, 0],  # 样本6-类别1
    [0, 1, 0],  # 样本7-类别1
    [0, 1, 0],  # 样本8-类别1
    [0, 0, 1],  # 样本9-类别2
    [0, 1, 0]  # 样本10-类别1
])

# 预测概率分数
y_score = np.array([
    [0.1, 0.2, 0.7],
    [0.1, 0.6, 0.3],
    [0.5, 0.2, 0.3],
    [0.1, 0.1, 0.8],
    [0.4, 0.2, 0.4],
    [0.6, 0.3, 0.1],
    [0.4, 0.2, 0.4],
    [0.4, 0.1, 0.5],
    [0.1, 0.1, 0.8],
    [0.1, 0.8, 0.1]
])
n_classes = y_true.shape[1]
fpr = dict()  # 每个类别的假阳性率
tpr = dict()  # 每个类别的真阳性率
roc_auc = dict()  # 每个类别的AUC值
for i in range(n_classes):
    # 计算ROC曲线点
    fpr_i, tpr_i, thresholds = roc_curve(y_true[:, i], y_score[:, i])
    if fpr_i[0] != 0.0:
        fpr_i = np.insert(fpr_i, 0, 0.0)
        tpr_i = np.insert(tpr_i, 0, 0.0)
    if fpr_i[-1] != 1.0:
        fpr_i = np.append(fpr_i, 1.0)
        tpr_i = np.append(tpr_i, 1.0)

    fpr[i] = fpr_i
    tpr[i] = tpr_i
    roc_auc[i] = auc(fpr[i], tpr[i])
# 3.1 Macro-average（宏平均）：平等对待每个类别
all_fpr = np.linspace(0, 1, 1000)
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
macro_auc = auc(all_fpr, mean_tpr)

# 3.2 Micro-average（微平均）：按样本数量加权，侧重大类
y_true_ravel = y_true.ravel()  # 展平真实标签
y_score_ravel = y_score.ravel()  # 展平预测分数
fpr_micro, tpr_micro, _ = roc_curve(y_true_ravel, y_score_ravel)
# 修正微平均曲线闭合
if fpr_micro[0] != 0.0:
    fpr_micro = np.insert(fpr_micro, 0, 0.0)
    tpr_micro = np.insert(tpr_micro, 0, 0.0)
if fpr_micro[-1] != 1.0:
    fpr_micro = np.append(fpr_micro, 1.0)
    tpr_micro = np.append(tpr_micro, 1.0)
micro_auc = auc(fpr_micro, tpr_micro)

# 3.3 Weighted-average（加权平均）：按类别样本数加权
weighted_tpr = np.zeros_like(all_fpr)
class_counts = y_true.sum(axis=0)  # 每个类别的样本数量
total_samples = len(y_true)

for i in range(n_classes):
    # 按类别样本占比加权，用等距FPR插值
    weight = class_counts[i] / total_samples
    weighted_tpr += weight * np.interp(all_fpr, fpr[i], tpr[i])
weighted_auc = auc(all_fpr, weighted_tpr)

# ===================== 4. 绘图 =====================
plt.figure(figsize=(10, 8))

# 绘制单类别ROC曲线
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, marker='o', markersize=4,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# 绘制三种平均ROC曲线
plt.plot(all_fpr, mean_tpr, color='red', lw=2, linestyle='--',
         label=f'Macro-average (AUC = {macro_auc:.2f})')
plt.plot(fpr_micro, tpr_micro, color='purple', lw=2, linestyle='-.',
         label=f'Micro-average (AUC = {micro_auc:.2f})')
plt.plot(all_fpr, weighted_tpr, color='brown', lw=2, linestyle=':',
         label=f'Weighted-average (AUC = {weighted_auc:.2f})')

# 绘制随机猜测基准线
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')

# 图表美化
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Multi-class ROC Curves (3 Average Methods)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)  # 添加网格线
# 显示图表
plt.show()