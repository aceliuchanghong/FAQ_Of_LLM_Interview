import ast
import timeit
import re
import tokenize
from io import StringIO


def evaluate_code(code: str, test_cases: list, verbose: bool = False) -> float:
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
        "runtime": 0.35,
        "brevity": 0.15,
        "correctness": 0.25,
        "readability": 0.05,
        "documentation": 0.05,
        "security": 0.15,
    }

    # 检查语法有效性
    try:
        ast.parse(code)
    except SyntaxError:
        if verbose:
            print("语法错误: -10")
        return -10.0  # 语法错误给予大幅负奖励

    # 提取函数名（假设代码包含单个函数定义）
    tree = ast.parse(code)
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    if not func_name:
        if verbose:
            print("未找到函数定义: -5")
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

    # 6. 安全性
    security_score = evaluate_security(code)

    if verbose:
        print(f"correctness_score: {correctness_score:.2f}")
        print(f"runtime_score: {runtime_score:.2f}")
        print(f"brevity_score: {brevity_score:.2f}")
        print(f"readability_score: {readability_score:.2f}")
        print(f"documentation_score: {documentation_score:.2f}")
        print(f"security_score: {security_score:.2f}")

    # 综合加权奖励
    total_reward = (
        weights["runtime"] * runtime_score
        + weights["brevity"] * brevity_score
        + weights["correctness"] * correctness_score
        + weights["readability"] * readability_score
        + weights["documentation"] * documentation_score
        + weights["security"] * security_score
    )

    return total_reward


def evaluate_correctness(code: str, func_name: str, test_cases: list) -> float:
    """评估代码是否通过测试用例 验证正确性和容错性。通过得1分,否则0分"""
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
    """
    评估代码的执行效率，运行时间越短越好
    将运行时间归一化为 0 到 1 之间的分数，公式为 `1 / (1 + time)`，时间越短，分数越高。
    """
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
    """
    评估代码的简洁性，代码行数越少越好。
    通过 AST 解析计算非空、非注释的代码行数。
    使用公式 `1 / (1 + lines / 10)` 将行数归一化为 0 到 1 之间的分数，行数越少，分数越高。
    """
    try:
        tree = ast.parse(code)
        # 获取所有节点的起始行号（忽略注释）
        code_lines = set()
        for node in ast.walk(tree):
            if hasattr(node, "lineno"):
                code_lines.add(node.lineno)

        # 原始代码行
        all_lines = code.split("\n")
        # 过滤空行
        non_empty_lines = [i + 1 for i, line in enumerate(all_lines) if line.strip()]
        # 仅保留非空且包含 AST 节点的行（排除注释）
        effective_lines = len([line for line in non_empty_lines if line in code_lines])

        # 归一化：行数越少越好
        return 1 / (1 + effective_lines / 10)
    except SyntaxError:
        return 0.0  # 语法错误返回 0


def evaluate_readability(code: str) -> float:
    """评估代码的可读性，基于标识符命名和代码结构复杂度。"""
    try:
        tree = ast.parse(code)

        # 1. 标识符命名评估
        identifiers = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        if not identifiers:
            return 0.5  # 无标识符时给中等分数

        # 计算平均标识符长度，鼓励适度长度（3-20 字符为最佳）
        avg_identifier_length = sum(len(id) for id in identifiers) / len(identifiers)
        length_score = min(
            max((avg_identifier_length - 3) / 17, 0), 1.0
        )  # 3-20 字符映射到 [0, 1]

        # 检查命名是否符合 PEP 8（小写加下划线）
        naming_convention_score = 1.0
        for id in identifiers:
            if isinstance(id, str) and id.isidentifier():
                # 函数名应为小写加下划线，变量名类似
                if any(c.isupper() for c in id):
                    naming_convention_score -= 0.1  # 包含大写扣分
                if not id.islower() and "_" not in id and len(id) > 1:
                    naming_convention_score -= 0.1  # 非下划线分隔扣分
        naming_convention_score = max(0.0, naming_convention_score)

        # 2. 代码结构复杂度（简单启发式：嵌套层数）
        max_depth = 0

        def calculate_depth(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(node):
                calculate_depth(child, depth + 1)

        calculate_depth(tree)
        complexity_score = 1 / (1 + max_depth / 5)  # 嵌套越深，分数越低

        # 3. 综合评分
        return (
            0.5 * length_score + 0.3 * naming_convention_score + 0.2 * complexity_score
        )
    except Exception:
        return 0.0


def evaluate_documentation(code: str) -> float:
    """
    评估文档字符串或注释的存在。
    评分规则：
        - 有文档字符串得 1.0 分。
        - 无文档字符串但有注释（单行或多行）得 0.5 分。
        - 无文档字符串且无注释得 0.0 分。
    """
    try:
        # 检查函数是否有文档字符串
        tree = ast.parse(code)
        has_docstring = any(
            isinstance(node, ast.FunctionDef) and ast.get_docstring(node)
            for node in ast.walk(tree)
        )
        if has_docstring:
            return 1.0

        # 使用 tokenize 检测所有注释（包括单行和多行）
        comment_count = 0
        try:
            tokens = tokenize.generate_tokens(StringIO(code).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:  # 单行注释
                    comment_count += 1
                elif token.type == tokenize.STRING:  # 检查多行注释（非文档字符串）
                    if token.string.startswith('"""') or token.string.startswith("'''"):
                        # 排除函数级文档字符串（已由 AST 处理）
                        if not any(
                            isinstance(node, ast.FunctionDef)
                            and token.start[0] == node.lineno
                            for node in ast.walk(tree)
                        ):
                            comment_count += 1
        except tokenize.TokenError:
            pass  # 忽略 tokenize 错误，继续评估

        return 0.5 if comment_count > 0 else 0.0
    except SyntaxError:
        return 0.0  # 语法错误返回 0


def evaluate_security(code_str: str) -> float:
    """
    对明显危险的模式进行初步的安全检查。
    真正的安全分析极其复杂。

    参数:
        code_str (str): Python 代码字符串。

    返回:
        float: 安全性得分 (0.0 到 1.0)。
    """
    score = 1.0
    # 检查 eval() 或 exec() 调用，如果与不受信任的输入一起使用可能很危险
    if re.search(r"\beval\s*\(", code_str) or re.search(r"\bexec\s*\(", code_str):
        score -= 0.75

    # 检查 pickle，它可能是远程代码执行 (RCE) 的一个途径
    if re.search(r"\bpickle\.load\b", code_str) or re.search(
        r"\bpickle\.loads\b", code_str
    ):
        score -= 0.5

    # 检查 os.system 或 subprocess 调用（简化检查 shell=True）
    if re.search(r"os\.system\s*\(", code_str) or re.search(
        r"subprocess\.(call|run|check_output|Popen)\s*\(.*shell\s*=\s*True", code_str
    ):
        score -= 0.5

    return max(0.0, score)


if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer
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
