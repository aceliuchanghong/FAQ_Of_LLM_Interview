import gym
import ast
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import List, Tuple
from typing import List, Tuple
import astunparse

from evaluate import evaluate_code


# 定义环境
class CodeOptimizationEnv(gym.Env):
    def __init__(self, initial_code: str, test_cases: List[Tuple]):
        super().__init__()
        self.initial_code = initial_code
        self.current_code = initial_code
        self.test_cases = test_cases
        # 动作空间：假设有 5 种 AST 转换操作（如简化循环、替换表达式等）
        self.action_space = gym.spaces.Discrete(5)
        # 状态空间：代码的特征向量（示例：AST 节点数、深度等）
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.max_steps = 50
        self.current_step = 0

    def seed(self, seed=None):
        """设置随机种子以确保可重复性"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_code = self.initial_code
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # 提取代码特征（如 AST 节点数、深度、token 数等）
        try:
            tree = ast.parse(self.current_code)
            node_count = sum(1 for _ in ast.walk(tree))
            depth = max(
                (len(list(ast.iter_child_nodes(n))) for n in ast.walk(tree)), default=0
            )
            return np.array(
                [node_count / 100, depth / 10, len(self.current_code.split()) / 100]
                + [0] * 7
            )  # 简化为 10 维
        except SyntaxError:
            return np.zeros(10)

    def step(self, action: int):
        # 应用 AST 转换
        self.current_code = self._apply_transformation(action)
        # 计算奖励
        reward = evaluate_code(self.current_code, self.test_cases)
        self.current_step += 1
        done = self.current_step >= self.max_steps or reward <= -5.0  # 提前终止
        return self._get_state(), reward, done, {}

    def _apply_transformation(self, action: int) -> str:
        try:
            tree = ast.parse(self.current_code)
            if action == 0:  # 示例：将 for 循环转为列表推导式
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
            return astunparse.unparse(tree).strip()
        except:
            return self.current_code


if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer
    export CUDA_VISIBLE_DEVICES=2
    python train.py

    pip install stable-baselines3 gym shimmy astunparse
    """

    initial_code = """
def add(a, b):
    return a + b
    """
    test_cases = [((1, 2), 3), ((0, 0), 0), ((-1, 1), 0)]

    # 创建环境
    env = make_vec_env(lambda: CodeOptimizationEnv(initial_code, test_cases), n_envs=1)

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",  # 使用简单的 MLP 策略
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # 训练
    model.learn(total_timesteps=10000)

    # 保存模型
    model.save("ppo_code_optimizer")

    # 测试优化
    env = CodeOptimizationEnv(initial_code, test_cases)
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Code: {env.current_code}, Reward: {reward}")
        if done:
            break
