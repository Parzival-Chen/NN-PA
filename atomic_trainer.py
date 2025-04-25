import random
import torch
from LearnerNN import LearnerNN
from PAModel import PAModel, combine_models,NotNetwork
from FormulaParser import FormulaParser, collect_atomic_formulas
from z3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保留之前的 atomic_trainer 辅助函数
def generate_data_for_formula(formula_str, var_list, num_samples=100, val_range=(0, 50)):
    z3_vars = {v: Int(v) for v in var_list}
    z3_formula = eval(formula_str, {'__builtins__': None}, z3_vars)
    pos, neg = [], []
    while len(pos) + len(neg) < num_samples:
        sample = [random.randint(*val_range) for _ in var_list]
        s = Solver()
        s.add(z3_formula)
        for var, val in zip(var_list, sample):
            s.add(z3_vars[var] == val)
        if s.check() == sat:
            pos.append(sample)
        else:
            neg.append(sample)
    return torch.tensor(pos, dtype=torch.float32), torch.tensor(neg, dtype=torch.float32)

def parse_atom_to_params(expr: str, var_list: list[str]) -> tuple[list[float], float]:
    lhs, rhs = expr.split('<')
    rhs = float(rhs.strip())
    lhs = lhs.replace('-', '+-')
    tokens = lhs.split('+')
    coeff_dict = {v: 0.0 for v in var_list}
    for token in tokens:
        token = token.strip().replace(' ', '')
        if not token:
            continue
        if '*' in token:
            parts = token.split('*')
            if len(parts) != 2:
                raise ValueError(f"Unrecognized token: {token}")
            coeff, var = parts
            coeff_dict[var] += float(coeff)
        elif token in var_list:
            coeff_dict[token] += 1.0
        elif token.startswith('-') and token[1:] in var_list:
            coeff_dict[token[1:]] -= 1.0
        else:
            raise ValueError(f"Unrecognized token: {token}")
    coeffs = [coeff_dict[v] for v in var_list]
    bias = -rhs
    return coeffs, bias

def train_atomic_model(expr: str, var_list: list[str], num_samples=100):
    coeffs, bias = parse_atom_to_params(expr, var_list)
    model = LearnerNN(None, coeffs, bias).to(device)
    pos_data, neg_data = generate_data_for_formula(expr, var_list, num_samples)

    pa_model = PAModel(var_list, expr, model,
                       pos_data=pos_data if len(pos_data) > 0 else None,
                       neg_data=neg_data if len(neg_data) > 0 else None)
    pa_model.train()
    return pa_model

def build_model_from_formula(
    formula_str: str,
    all_vars: list[str],
    initial_num_samples: int = 100,
    retrain_num_samples: int = 100,
    acc_threshold: float = 0.9,
    max_retrain_rounds: int = 10
) -> PAModel:
    parser = FormulaParser(formula_str)
    tree = parser.parse()

    def build(node) -> PAModel:
        if not node.children:  # 原子命题
            print(f" 训练原子命题: {node.value}")
            used_vars = [v for v in all_vars if v in node.value]
            try:
                model = train_atomic_model(node.value, used_vars, num_samples=initial_num_samples)
                train_until_good(model, node.value)
                return model
            except Exception as e:
                print(f"❌ 错误: 原子命题 {node.value} 训练失败，原因: {e}")
                raise RuntimeError(f"训练原子命题 {node.value} 失败") from e

        if node.value == 'And':
            print(f"组合逻辑: And")
            left = build(node.children[0])
            right = build(node.children[1])
            if left is None or right is None:
                raise RuntimeError(f"组合And节点 {node.value} 时，子节点构建失败")
            model = combine_models(left, right, all_vars, num_samples=initial_num_samples, logic_type='and')
            train_until_good(model, f"And({left.formula_str}, {right.formula_str})")
            return model

        if node.value == 'Or':
            print(f"组合逻辑: Or")
            left = build(node.children[0])
            right = build(node.children[1])
            if left is None or right is None:
                raise RuntimeError(f"组合Or节点 {node.value} 时，子节点构建失败")
            model = combine_models(left, right, all_vars, num_samples=initial_num_samples, logic_type='or')
            train_until_good(model, f"Or({left.formula_str}, {right.formula_str})")
            return model

        if node.value == 'Not':
            print(f"⚙️ 组合逻辑: Not")
            child = build(node.children[0])
            if child is None:
                raise RuntimeError(f"组合Not节点 {node.value} 时，子节点构建失败")
            if isinstance(child.model, NotNetwork):
                print(f"✨ 检测到连续Not，简化！")
                simplified_model = PAModel(
                    child.vars,
                    child.formula_str,
                    child.model,
                    child.pos_data,
                    child.neg_data
                )
                return simplified_model

            model = combine_models(child, None, all_vars, num_samples=initial_num_samples, logic_type='not')
            train_until_good(model, f"Not({child.formula_str})")
            return model

        raise ValueError(f"Unknown node value: {node.value}")

    # 测试 + 反例回馈机制
    def train_until_good(model: PAModel, model_name: str):
        print(f"\n 开始验证节点: {model_name}")
        for round_idx in range(max_retrain_rounds):
            correct = 0
            false_pos, false_neg = [], []
            total = 100
            test_samples = [[random.randint(0, 50) for _ in model.vars] for _ in range(total)]  # 范围改成0-50

            for x in test_samples:
                pred = model.predict(x)
                truth = model.verify(x)
                if pred == truth:
                    correct += 1
                else:
                    if truth:
                        false_neg.append(x)
                    else:
                        false_pos.append(x)

            acc = correct / total
            print(f" 节点 {model_name} 当前测试准确率: {acc:.2%}")

            if acc >= acc_threshold:
                print(f"节点 {model_name} 达到要求准确率 {acc_threshold:.2%}，停止训练。\n")
                return
            else:
                print(f" 节点 {model_name} 准确率不足，进行第 {round_idx + 1} 次反例反馈训练...")
                model.update(false_neg, false_pos)

        print(f" 节点 {model_name} 经过 {max_retrain_rounds} 次训练仍未达到目标准确率。\n")

    final_model = build(tree)
    return final_model

