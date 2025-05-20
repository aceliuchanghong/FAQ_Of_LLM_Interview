import ast
import timeit
import unittest
import re
from typing import Callable, Any


def evaluate_code(code: str, test_cases: list) -> float:
    """
    评估Python代码片段,基于多个指标计算奖励。

    参数：
        code (str):Python代码片段。
        test_cases (list):测试用例列表，格式为[(输入参数, 期望输出), ...]。

    返回：
        综合奖励分数。
    """
    # 定义各指标权重
    weights = {
        "runtime": 0.4,
        "brevity": 0.2,
        "correctness": 0.3,
        "readability": 0.05,
        "documentation": 0.05,
    }

    # 检查语法有效性
    try:
        ast.parse(code)
    except SyntaxError:
        return -10.0  # 语法错误给予大幅负奖励

    # 提取函数名（假设代码包含单个函数定义）
    tree = ast.parse(code)
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    if not func_name:
        return -5.0  # 无函数定义时惩罚

    # 1. 正确性和容错性
    correctness_score = evaluate_correctness(code, func_name, test_cases)
    if correctness_score == 0:
        return -5.0  # 测试失败时惩罚

    # 2. 运行时间
    runtime_score = evaluate_runtime(code, func_name, test_cases)

    # 3. 简洁性
    brevity_score = evaluate_brevity(code)

    # 4. 可读性
    readability_score = evaluate_readability(code)

    # 5. 文档完整性
    documentation_score = evaluate_documentation(code)

    # 综合加权奖励
    total_reward = (
        weights["runtime"] * runtime_score
        + weights["brevity"] * brevity_score
        + weights["correctness"] * correctness_score
        + weights["readability"] * readability_score
        + weights["documentation"] * documentation_score
    )

    return total_reward


def evaluate_correctness(code: str, func_name: str, test_cases: list) -> float:
    """评估代码是否通过测试用例，验证正确性和容错性。"""
    try:
        # 在安全命名空间中执行代码
        namespace = {}
        exec(code, namespace)
        func = namespace.get(func_name)
        if not callable(func):
            return 0.0

        # 运行测试用例
        for args, expected in test_cases:
            try:
                result = func(*args) if isinstance(args, tuple) else func(args)
                if result != expected:
                    return 0.0
            except Exception:
                return 0.0  # 运行时错误（包括容错性失败）给予0分
        return 1.0
    except Exception:
        return 0.0


def evaluate_runtime(code: str, func_name: str, test_cases: list) -> float:
    """使用timeit评估代码执行时间。"""
    try:
        setup = f"{code}\nfunc = {func_name}"
        test_input = test_cases[0][0]  # 使用第一个测试用例进行计时
        stmt = (
            f"func({test_input})"
            if not isinstance(test_input, tuple)
            else f"func{test_input}"
        )
        time = timeit.timeit(stmt=stmt, setup=setup, number=1000)
        # 归一化：运行时间越短越好
        return 1 / (1 + time)
    except Exception:
        return 0.0


def evaluate_brevity(code: str) -> float:
    """基于代码行数评估简洁性。"""
    lines = len(
        [
            line
            for line in code.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
    )
    # 归一化：行数越少越好
    return 1 / (1 + lines / 10)


def evaluate_readability(code: str) -> float:
    """基于标识符长度和缩进一致性评估可读性。"""
    tree = ast.parse(code)
    identifiers = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    avg_identifier_length = sum(len(id) for id in identifiers) / (len(identifiers) + 1)
    # 检查缩进一致性（简单启发式）
    lines = code.split("\n")
    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    indent_variance = sum((x - 4) ** 2 for x in indent_levels) / (
        len(indent_levels) + 1
    )
    # 综合：较长的标识符和一致的缩进更好
    return min(avg_identifier_length / 10, 1.0) * (1 / (1 + indent_variance))


def evaluate_documentation(code: str) -> float:
    """评估文档字符串或注释的存在。"""
    tree = ast.parse(code)
    has_docstring = any(
        isinstance(node, ast.FunctionDef) and ast.get_docstring(node)
        for node in ast.walk(tree)
    )
    comment_count = len(re.findall(r"#.*", code))
    return 1.0 if has_docstring else (0.5 if comment_count > 0 else 0.0)


if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer/
    python evaluate.py
    """
    example_code = """
def fibonacci(n):
    '''计算第n个斐波那契数。'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    """
    test_cases = [(0, 0), (1, 1), (5, 5), (10, 55)]
    reward = evaluate_code(example_code, test_cases)
    print(f"奖励分数: {reward}")
