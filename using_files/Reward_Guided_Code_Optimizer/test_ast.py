import ast
import astunparse


# 2. 定义AST变换器
class ForLoopTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        new_body = []
        for_loop_detected = False
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "total"
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == 0
            ):
                for_loop_detected = True
                continue  # 跳过 total = 0

            # 检查是否是for循环
            if (
                for_loop_detected
                and isinstance(stmt, ast.For)
                and isinstance(stmt.body[0], ast.AugAssign)
                and isinstance(stmt.body[0].target, ast.Name)
                and stmt.body[0].target.id == "total"
            ):
                new_stmt = ast.Assign(
                    targets=[ast.Name(id="total", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="sum", ctx=ast.Load()),
                        args=[stmt.iter],
                        keywords=[],
                    ),
                )
                new_body.append(new_stmt)
                for_loop_detected = False
                continue

            new_body.append(stmt)

        node.body = new_body
        return self.generic_visit(node)  # 继续遍历子节点


if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer/
    python test_ast.py
    """
    code = """
def sum_list(lst):
    total = 0
    for i in lst:
        total += i
    return total
"""

    # 1. 解析AST
    tree = ast.parse(code)

    # 3. 执行转换
    transformer = ForLoopTransformer()
    new_tree = transformer.visit(tree)

    # 4. 转换回代码
    new_code = astunparse.unparse(new_tree)
    print(new_code)
