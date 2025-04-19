import torch
import torch.nn as nn
from LearnerNN import LearnerNN
from PAModel import PAModel
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

from atomic_trainer import train_atomic_model
from PAModel import combine_and_model

M1, _ = train_atomic_model("2*x + 3*y - 5*z < 1", ['x', 'y', 'z'])
M2, _ = train_atomic_model("2*y + z -w < 3", ['y', 'z', 'w'])
M3 = combine_and_model(M1, M2, var_order=['x', 'y', 'z', 'w'])
M3.train(epochs=100)

# 第一轮评估
print("\n 正在评估组合模型 M3 的预测准确率...")
correct = 0
false_pos, false_neg = [], []
total = 100
test_samples = [[random.randint(0, 10) for _ in M3.vars] for _ in range(total)]

for x in test_samples:
    pred = M3.predict(x)
    truth = M3.verify(x)
    if pred == truth:
        correct += 1
    else:
        print(f" 错误样例: 输入={x}, 预测={pred}, 实际={truth}")
        if truth:
            false_neg.append(x)
        else:
            false_pos.append(x)

acc = correct / total
print(f"\n 第一轮 M3 测试集准确率: {acc:.2%}")

# 加入反例重新训练
print("\n 将反例加入训练集并重新训练 M3 ...")
M3.update(false_neg, false_pos)

# 第二轮评估
print("\n 第二轮测试开始 ...")
correct2 = 0
false_pos2, false_neg2 = [], []
test_samples2 = [[random.randint(0, 10) for _ in M3.vars] for _ in range(total)]

for x in test_samples2:
    pred = M3.predict(x)
    truth = M3.verify(x)
    if pred == truth:
        correct2 += 1
    else:
        print(f" 第二轮错误样例: 输入={x}, 预测={pred}, 实际={truth}")
        if truth:
            false_neg2.append(x)
        else:
            false_pos2.append(x)

acc2 = correct2 / total
print(f"\n 第二轮 M3 测试集准确率: {acc2:.2%}")

# 加入第二轮反例继续训练
print("\n 将第二轮反例加入训练集并再次训练 M3 ...")
M3.update(false_neg2, false_pos2)

# 第三轮评估
print("\n 第三轮测试开始 ...")
correct3 = 0
test_samples3 = [[random.randint(0, 10) for _ in M3.vars] for _ in range(total)]

for x in test_samples3:
    pred = M3.predict(x)
    truth = M3.verify(x)
    if pred == truth:
        correct3 += 1
    else:
        print(f" 第三轮错误样例: 输入={x}, 预测={pred}, 实际={truth}")

acc3 = correct3 / total
print(f"\n 第三轮 M3 测试集准确率: {acc3:.2%}")

# TODO: 在LearnerNN.py里添加一个逻辑与方法, 接收两个网络输出逻辑与网络。初步想法：ensemble knowledge distillation