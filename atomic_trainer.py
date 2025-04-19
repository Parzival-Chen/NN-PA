import random
import torch
from LearnerNN import LearnerNN
from PAModel import PAModel
from z3 import *
from FormulaParser import FormulaParser, collect_atomic_formulas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_data_for_formula(formula_str, var_list, num_samples=100, val_range=(0, 10)):
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

    # 直接跳过训练阶段（原子神经元已固定）
    print(f" 已构建原子命题模型: \"{expr}\"（不训练）")
    def evaluate(model, pos_data, neg_data):
        correct = 0
        for sample in pos_data:
            if model.predict(sample.tolist()):
                correct += 1
        for sample in neg_data:
            if not model.predict(sample.tolist()):
                correct += 1
        acc = correct / (len(pos_data) + len(neg_data))
        return acc

    acc = evaluate(pa_model, pos_data, neg_data)
    print(f" 原子命题 \"{expr}\" 静态准确率: {acc:.2%}")
    return pa_model, acc
def build_atomic_models_from_formula(formula_str: str, all_vars: list[str], num_samples=100):
    parser = FormulaParser(formula_str)
    tree = parser.parse()
    atoms = collect_atomic_formulas(tree)
    model_dict = {}
    for atom in atoms:
        used_vars = [v for v in all_vars if v in atom]
        model, acc = train_atomic_model(atom, used_vars, num_samples=num_samples)
        model_dict[atom] = (model, acc)
        print(f" 已构建原子命题: \"{atom}\" → 准确率: {acc:.2%}")
    return model_dict
if __name__ == "__main__":
    formula = "Or(And(2*x + 3*y < 5, Not(z - w < 3)), x + y < 6)"
    all_vars = ["x", "y", "z", "w"]
    atom_models = build_atomic_models_from_formula(formula, all_vars)
    print("\n 所有原子模型构建完毕：")
    for atom, (model, acc) in atom_models.items():
        print(f"  - {atom} → 准确率: {acc:.2%}")
