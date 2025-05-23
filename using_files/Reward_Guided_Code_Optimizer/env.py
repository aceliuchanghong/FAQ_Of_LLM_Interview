import gym
import ast
import numpy as np
import astunparse
from evaluate import evaluate_code


class CodeOptimizationEnv(gym.Env):
    def __init__(self, initial_code: str, test_cases: list):
        super().__init__()
        self.initial_code = initial_code
        self.current_code = initial_code
        self.test_cases = test_cases
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.max_steps = 50
        self.current_step = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        """重置，返回初始状态"""
        self.current_code = self.initial_code
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        try:
            tree = ast.parse(self.current_code)
            node_count = sum(1 for _ in ast.walk(tree))
            depth = max(
                (len(list(ast.iter_child_nodes(n))) for n in ast.walk(tree)), default=0
            )
            return np.array(
                [node_count / 100, depth / 10, len(self.current_code.split()) / 100]
                + [0] * 7
            )
        except SyntaxError:
            return np.zeros(10)

    def step(self, action: int):
        """执行一步动作，返回当前状态"""
        self.current_code = self._apply_transformation(action)
        reward = evaluate_code(self.current_code, self.test_cases)
        self.current_step += 1
        done = self.current_step >= self.max_steps or reward <= -5.0
        return self._get_state(), reward, done, {}

    def _apply_transformation(self, action: int) -> str:
        try:
            tree = ast.parse(self.current_code)
            if action == 0:
                for node in ast.walk(tree):
                    if isinstance(node, ast.For):
                        if (
                            isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Call)
                            and isinstance(node.body[0].value.func, ast.Attribute)
                            and node.body[0].value.func.attr == "append"
                        ):
                            new_listcomp = ast.ListComp(
                                elt=node.body[0].value.args[0],
                                generators=[
                                    ast.comprehension(
                                        target=node.target, iter=node.iter, ifs=[]
                                    )
                                ],
                            )
                            assign = ast.Assign(
                                targets=[ast.Name(id="lst", ctx=ast.Store())],
                                value=new_listcomp,
                            )
                            tree.body.append(assign)
                            break
            # TODO: Implement actions 1-4 for other transformations
            return astunparse.unparse(tree).strip()
        except:
            return self.current_code
