import re
from typing import Union
# 定义语法树的节点类型
class FormulaNode:
    def __init__(self, value: str, children=None):
        self.value = value
        self.children: list['FormulaNode'] = children or []

    def __repr__(self, level=0):
        indent = '  ' * level
        result = f"{indent}{self.value}\n"
        for child in self.children:
            result += child.__repr__(level + 1)
        return result
LOGIC_FUNCS = {"And", "Or", "Not", "Implies"}
# 解析函数
class FormulaParser:
    def __init__(self, formula_str: str):
        self.tokens = self.tokenize(formula_str)
        self.index = 0

    def tokenize(self, formula_str: str) -> list[str]:
        """
        将公式字符串分割为 token 序列，保留括号和操作符
        支持格式：And(...), Or(...), Not(...), Implies(...), 任意原子不等式
        """
        token_pattern = r"\(|\)|,|<=|>=|==|!=|<|>|\w+|[+\-*/]"
        tokens = re.findall(token_pattern, formula_str)
        return tokens

    def parse(self) -> FormulaNode:
        return self._parse_expr()
    def _parse_expr(self) -> FormulaNode:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of formula")
        token = self.tokens[self.index]
        if token in LOGIC_FUNCS:
            self.index += 1  # 跳过函数名
            self._consume('(')
            children = []
            while True:
                children.append(self._parse_expr())
                if self.tokens[self.index] == ',':
                    self.index += 1  # consume comma
                else:
                    break
            self._consume(')')
            return FormulaNode(token, children)
        else:
            # 解析原子命题（如 x + y < 5）
            atom = []
            while self.index < len(self.tokens) and self.tokens[self.index] not in {',', ')'}:
                atom.append(self.tokens[self.index])
                self.index += 1
            return FormulaNode(' '.join(atom))
    def _consume(self, expected_token):
        if self.tokens[self.index] != expected_token:
            raise ValueError(f"Expected '{expected_token}', got '{self.tokens[self.index]}'")
        self.index += 1


# 提取所有原子命题（语法树叶子节点）
def collect_atomic_formulas(root: FormulaNode) -> list[str]:
    """
    递归收集语法树中的所有叶子节点（原子命题）
    """
    if not root.children:
        return [root.value]
    atoms = []
    for child in root.children:
        atoms.extend(collect_atomic_formulas(child))
    return atoms
# 示例用法
if __name__ == '__main__':
    formula = "Or(And(2*x + 3*y < 5, Not(z - w < 3)),x+y<6)"
    parser = FormulaParser(formula)
    tree = parser.parse()
    print(" 解析树结构:")
    print(tree)
    print(" 提取的原子命题:")
    atoms = collect_atomic_formulas(tree)
    for i, atom in enumerate(atoms):
        print(f"  [{i + 1}] {atom}")
