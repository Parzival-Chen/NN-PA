import torch
import random
from LearnerNN import LearnerNN
from PAModel import PAModel, combine_models
from atomic_trainer import train_atomic_model, build_model_from_formula
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
def generate_random_formula(vars, max_constant=50, max_depth=5):
    if max_depth == 0:
        # 叶子节点：生成一个原子不等式
        num_vars_in_atom = random.randint(2, len(vars))  # 随机使用2到所有变量
        chosen_vars = random.sample(vars, num_vars_in_atom)

        terms = []
        for var in chosen_vars:
            coeff = random.randint(1, 10)
            terms.append(f"{coeff}*{var}")

        lhs = " + ".join(terms)
        constant = random.randint(0, max_constant)
        return f"{lhs} < {constant}"

    # 内部节点：随机选逻辑操作符
    op = random.choice(["And", "Or", "Not"])
    if op == "Not":
        subformula = generate_random_formula(vars, max_constant, max_depth-1)
        return f"Not({subformula})"
    else:
        left = generate_random_formula(vars, max_constant, max_depth-1)
        right = generate_random_formula(vars, max_constant, max_depth-1)
        return f"{op}({left}, {right})"

# =================== 这里开始测试 ===================

all_vars = ["x", "y", "z", "w"]

formula = generate_random_formula(all_vars, max_constant=500, max_depth=3)
print(f"\n 随机生成公式: {formula}\n")
final_model = build_model_from_formula(
    formula,
    all_vars,
    initial_num_samples=100,
    retrain_num_samples=100,
    acc_threshold=0.9,
    max_retrain_rounds=10
)
