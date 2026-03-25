import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc  # 用来算AUC

plt.switch_backend('TkAgg')

pp=[['T',0.9],['F',0.4],['F',0.2],['T',0.6],
    ['F',0.5],['F',0.5],['T',0.7],['T',0.4]]
aa=[0.9,0.8,0.7,0.6,0.56,0.55,0.54,0.53,0.52,0.51,0.505,0.4,0.39,0.38,0.37,0.36,0.35,0.34,0.33,0.32,0.3,0.1]
recall =[]
precision =[]
TPR =[]
FPR =[]

for a in aa:
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    x = 0
    y = 0

    for p in pp:
        if( p[0] == 'T') and (p[1]>=a):
            tp = tp + 1
        elif( p[0] == 'T') and (p[1]<a):
            fn = fn + 1
        elif( p[0] == 'F') and (p[1]>=a):
            fp = fp +1
        elif( p[0] == 'F') and (p[1]<a):
            tn =tn + 1

    x = float(tp)/float(tp+fn)
    y = float(tp)/float(tp+fp) if (tp+fp) > 0 else 0.0
    fpr =float(fp)/float(fp+tn) if (fp+tn) > 0 else 0.0

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)


auc_score = auc(FPR, TPR)
print(f"AUC = {auc_score:.4f}")  # 控制台输出


plt.close('all')
plt.figure(figsize=(5,5))
plt.title(f'ROC curve (AUC = {auc_score:.4f})', fontsize=16)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.show()