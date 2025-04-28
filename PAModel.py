import torch
import torch.nn as nn
import random
from z3 import *
class AlwaysTrueModel(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self._target_device = device

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.ones(batch_size, 1, device=x.device)

    def predict(self, inputs):
        return True

class AlwaysFalseModel(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self._target_device = device

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 1, device=x.device)

    def predict(self, inputs):
        return False


class PAModel:
    def __init__(self, vars, formula_str, model, pos_data=None, neg_data=None):
        self.vars = vars
        self.formula_str = formula_str
        self.z3_vars = {v: Int(v) for v in vars}
        self.z3_formula = eval(formula_str, {
            '__builtins__': None,
            'And': And, 'Or': Or, 'Not': Not, 'Implies': Implies
        }, self.z3_vars)

        if pos_data is not None:
            if not isinstance(pos_data, torch.Tensor):
                pos_data = torch.tensor(pos_data, dtype=torch.float32)
            if pos_data.ndim == 0 or pos_data.shape[0] == 0:
                pos_data = None
        self.pos_data = pos_data

        if neg_data is not None:
            if not isinstance(neg_data, torch.Tensor):
                neg_data = torch.tensor(neg_data, dtype=torch.float32)
            if neg_data.ndim == 0 or neg_data.shape[0] == 0:
                neg_data = None
        self.neg_data = neg_data

        self.model = model

    def train(self, epochs=100):
        if hasattr(self.model, 'is_fixed') and self.model.is_fixed:
            print(f"模型 {self.formula_str} 是固定原子神经元，跳过训练")
            return
        if self.pos_data is None or self.neg_data is None:
            raise ValueError("Training data not provided or is empty")
        if len(self.pos_data) == 0 or len(self.neg_data) == 0:
            print("Warning: One of the training datasets is empty.")
            return
        X = torch.cat([self.pos_data, self.neg_data])
        y = torch.cat([
            torch.ones(len(self.pos_data)),
            torch.zeros(len(self.neg_data))
        ]).unsqueeze(1)
        device = next(self.model.parameters()).device
        X = X.to(device)
        y = y.to(device)

        if hasattr(self.model, 'train_on_data'):
            self.model.train_on_data(X, y, epochs)
            return

        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

    def predict(self, inputs):
        inputs = [int(i) for i in inputs]
        return self.model.predict(inputs)

    def verify(self, inputs):
        inputs = [int(i) for i in inputs]
        s = Solver()
        s.add(self.z3_formula)
        for var, val in zip(self.vars, inputs):
            s.add(self.z3_vars[var] == val)
        return s.check() == sat

    def add_example(self, example, is_positive=True):
        tensor = torch.tensor(example, dtype=torch.float32).unsqueeze(0)
        if is_positive:
            self.pos_data = torch.cat([self.pos_data, tensor]) if self.pos_data is not None else tensor
        else:
            self.neg_data = torch.cat([self.neg_data, tensor]) if self.neg_data is not None else tensor

    def update(self, new_pos_data, new_neg_data):
        new_pos_tensor = torch.tensor(new_pos_data, dtype=torch.float32) if new_pos_data and len(new_pos_data) > 0 else None
        new_neg_tensor = torch.tensor(new_neg_data, dtype=torch.float32) if new_neg_data and len(new_neg_data) > 0 else None

        if new_pos_tensor is not None:
            self.pos_data = torch.cat([self.pos_data, new_pos_tensor]) if self.pos_data is not None else new_pos_tensor
        if new_neg_tensor is not None:
            self.neg_data = torch.cat([self.neg_data, new_neg_tensor]) if self.neg_data is not None else new_neg_tensor

        self.train()

    def save_config(self, file_path, save_weights=False):
        self.model.save_config(file_path, save_weights)


class AndNetwork(nn.Module):
    def __init__(self, model1: PAModel, model2: PAModel, var_order: list[str]):
        super().__init__()
        self.model1 = model1.model
        self.model2 = model2.model
        self.vars1 = model1.vars
        self.vars2 = model2.vars
        self.var_order = var_order

        self.classifier = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[:, [self.var_order.index(v) for v in self.vars1]]
        x2 = x[:, [self.var_order.index(v) for v in self.vars2]]

        out1 = self.model1(x1)
        if isinstance(out1, tuple):
            out1 = out1[0]

        out2 = self.model2(x2)
        if isinstance(out2, tuple):
            out2 = out2[0]

        logic_input = torch.cat([out1, out2], dim=1)
        combined = self.classifier(logic_input)
        return combined, out1, out2

    def predict(self, inputs):
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(next(self.classifier.parameters()).device)
            pred, _, _ = self.forward(x)
            return pred.item() > 0.5

    def train_on_data(self, X, y, epochs=100, alpha=0.5):
        optimizer = torch.optim.Adam(self.parameters())
        loss_fn = nn.BCELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred, out1, out2 = self.forward(X)
            loss_z3 = loss_fn(pred, y)
            logic_truth = ((out1 > 0.5) & (out2 > 0.5)).float()
            distill_loss = loss_fn(pred, logic_truth)
            loss = loss_z3 + alpha * distill_loss
            loss.backward()
            optimizer.step()


class OrNetwork(nn.Module):
    def __init__(self, model1: PAModel, model2: PAModel, var_order: list[str]):
        super().__init__()
        self.model1 = model1.model
        self.model2 = model2.model
        self.vars1 = model1.vars
        self.vars2 = model2.vars
        self.var_order = var_order

        self.classifier = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[:, [self.var_order.index(v) for v in self.vars1]]
        x2 = x[:, [self.var_order.index(v) for v in self.vars2]]

        out1 = self.model1(x1)
        if isinstance(out1, tuple):
            out1 = out1[0]

        out2 = self.model2(x2)
        if isinstance(out2, tuple):
            out2 = out2[0]

        logic_input = torch.cat([out1, out2], dim=1)
        combined = self.classifier(logic_input)
        return combined, out1, out2

    def predict(self, inputs):
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(next(self.classifier.parameters()).device)
            pred, _, _ = self.forward(x)
            return pred.item() > 0.5

    def train_on_data(self, X, y, epochs=100, alpha=0.5):
        optimizer = torch.optim.Adam(self.parameters())
        loss_fn = nn.BCELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred, out1, out2 = self.forward(X)
            loss_z3 = loss_fn(pred, y)
            logic_truth = ((out1 > 0.5) | (out2 > 0.5)).float()
            distill_loss = loss_fn(pred, logic_truth)
            loss = loss_z3 + alpha * distill_loss
            loss.backward()
            optimizer.step()


class NotNetwork(nn.Module):
    def __init__(self, model: PAModel, var_order: list[str]):
        super().__init__()
        self.model = model.model
        self.vars = model.vars
        self.var_order = var_order

        # ✨ 新增！顶上添加一个小线性层
        self.top_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_sub = x[:, [self.var_order.index(v) for v in self.vars]]
        out = self.model(x_sub)
        if isinstance(out, tuple):
            out = out[0]

        flipped = 1.0 - out  # 取反
        return self.top_layer(flipped), flipped  # 用顶上的可训练层做输出

    def predict(self, inputs):
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(next(self.top_layer.parameters()).device)
            pred, _ = self.forward(x)
            return pred.item() > 0.5

    def verify(self, inputs):
        return not self.model.verify(inputs)

    def train_on_data(self, X, y, epochs=20):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred, _ = self.forward(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()



def combine_models(M1: PAModel, M2: PAModel = None, var_order: list[str] = None, num_samples=100, logic_type="and") -> PAModel:
    def get_device_safe(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            if hasattr(model, '_target_device'):
                return model._target_device
            else:
                return torch.device('cpu')

    def sample_data(z3_formula, z3_vars, var_order, num_samples, logic_type_name):
        pos, neg = [], []
        attempts = 0
        while attempts < 10 * num_samples:
            sample = [random.randint(0, 50) for _ in var_order]
            s = Solver()
            s.add(z3_formula)
            for v, val in zip(var_order, sample):
                s.add(z3_vars[v] == val)
            if s.check() == sat:
                pos.append(sample)
            else:
                neg.append(sample)
            attempts += 1

            total = len(pos) + len(neg)
            if total >= num_samples:
                ratio = len(pos) / total
                if 0.2 <= ratio <= 0.8:
                    break  # 满足比例要求了，可以停止

        # ⚡ 补充极端样本
        if len(pos) == 0 or len(neg) == 0:
            print(f"⚡ 尝试补充小值和大值样本 ({logic_type_name})")
            special_samples = []
            for value in [0, 1, 49, 50]:
                special_samples.append([value] * len(var_order))
                for i in range(len(var_order)):
                    temp = [50] * len(var_order)
                    temp[i] = value
                    special_samples.append(temp)

            for special in special_samples:
                s = Solver()
                s.add(z3_formula)
                for v, val in zip(var_order, special):
                    s.add(z3_vars[v] == val)
                if s.check() == sat:
                    pos.append(special)
                else:
                    neg.append(special)

        # ✨ 极端样本补充后再判断
        if len(pos) == 0:
            print(f"⚡ 检测到公式恒False，直接生成AlwaysFalseModel ({logic_type_name})")
            model = AlwaysFalseModel()
            return 'false', model
        if len(neg) == 0:
            print(f"⚡ 检测到公式恒True，直接生成AlwaysTrueModel ({logic_type_name})")
            model = AlwaysTrueModel()
            return 'true', model

        pos_tensor = torch.tensor(pos, dtype=torch.float32)
        neg_tensor = torch.tensor(neg, dtype=torch.float32)
        return pos_tensor, neg_tensor

    def handle_sample_result(sample_result, var_order, combined_formula_str):
        if isinstance(sample_result, tuple) and isinstance(sample_result[0], str):
            special_case, model = sample_result
            if special_case == 'true':
                pos_tensor = torch.ones((10, len(var_order)), dtype=torch.float32)
                neg_tensor = None
            else:
                pos_tensor = None
                neg_tensor = torch.ones((10, len(var_order)), dtype=torch.float32)
            return PAModel(var_order, combined_formula_str, model, pos_tensor, neg_tensor)
        else:
            pos_tensor, neg_tensor = sample_result
            return pos_tensor, neg_tensor

    if logic_type == "and":
        if M2 is None:
            raise ValueError("AND 逻辑需要提供两个子模型 M1 和 M2")
        z3_vars = {v: Int(v) for v in var_order}
        z3_formula = And(M1.z3_formula, M2.z3_formula)
        combined_formula_str = f"And({M1.formula_str}, {M2.formula_str})"
        model = AndNetwork(M1, M2, var_order).to(get_device_safe(M1.model))


        sample_result = sample_data(z3_formula, z3_vars, var_order, num_samples, "And")
        handled = handle_sample_result(sample_result, var_order, combined_formula_str)
        if isinstance(handled, PAModel):
            return handled
        pos_tensor, neg_tensor = handled
        return PAModel(var_order, combined_formula_str, model, pos_tensor, neg_tensor)

    elif logic_type == "or":
        if M2 is None:
            raise ValueError("OR 逻辑需要提供两个子模型 M1 和 M2")
        z3_vars = {v: Int(v) for v in var_order}
        z3_formula = Or(M1.z3_formula, M2.z3_formula)
        combined_formula_str = f"Or({M1.formula_str}, {M2.formula_str})"
        model = OrNetwork(M1, M2, var_order).to(get_device_safe(M1.model))


        sample_result = sample_data(z3_formula, z3_vars, var_order, num_samples, "Or")
        handled = handle_sample_result(sample_result, var_order, combined_formula_str)
        if isinstance(handled, PAModel):
            return handled
        pos_tensor, neg_tensor = handled
        return PAModel(var_order, combined_formula_str, model, pos_tensor, neg_tensor)

    elif logic_type == "not":
        if var_order is None:
            var_order = M1.vars
        z3_vars = {v: Int(v) for v in var_order}
        z3_formula = Not(M1.z3_formula)
        combined_formula_str = f"Not({M1.formula_str})"
        model = NotNetwork(M1, var_order).to(get_device_safe(M1.model))
        sample_result = sample_data(z3_formula, z3_vars, var_order, num_samples, "Not")
        handled = handle_sample_result(sample_result, var_order, combined_formula_str)
        if isinstance(handled, PAModel):
            return handled
        pos_tensor, neg_tensor = handled
        return PAModel(var_order, combined_formula_str, model, pos_tensor, neg_tensor)

    else:
        raise ValueError(f"Unsupported logic_type: {logic_type}")




    # def combine_presburger_models(M1, M2):
    #     """
    #     合并两个PresburgerModel对象，处理共享变量y,z
    #     :param M1: 含变量x,y,z的模型
    #     :param M2: 含变量y,z,w的模型
    #     :return: 含变量x,y,z,w的新模型M3
    #     """
    #     # 合并变量（保持共享变量y,z的一致性）
    #     new_vars = ['x'] + sorted(list(set(M1.variables) & set(M2.variables))) + ['w']  # ['x','y','z','w']

    #     # 构建新公式（示例使用逻辑与组合，实际可根据需求修改）
    #     new_formula = f"And({M1.formula_str}, {M2.formula_str})"

    #     # 构建集成神经网络（使用PyTorch的ModuleList）
    #     class IntegratedModel(nn.Module):
    #         def __init__(self, model1, model2):
    #             super().__init__()
    #             self.model1 = model1.model
    #             self.model2 = model2.model
    #             self.combine = nn.Linear(2, 1)  # 集成两个模型的输出

    #         def forward(self, x):
    #             # 分割输入：x[0]对应x, x[1:3]对应y,z, x[3]对应w
    #             out1 = self.model1(x[[0,1,2]])  # M1处理x,y,z
    #             out2 = self.model2(x[[1,2,3]])  # M2处理y,z,w
    #             return torch.sigmoid(self.combine(torch.cat([out1, out2], dim=1)))

    #     # 创建新模型实例
    #     integrated_nn = IntegratedModel(M1, M2)
    #     return PAModel(integrated_nn, new_formula, new_vars)