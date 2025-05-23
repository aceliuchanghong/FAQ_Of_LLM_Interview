from env import CodeOptimizationEnv
from agent import CodeOptimizationAgent
from stable_baselines3 import PPO
import numpy as np
import random

if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer
    python inference.py
    """
    random.seed(42)
    np.random.seed(42)

    # 定义初始代码和测试用例
    initial_code = """
def sum_list(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
"""
    test_cases = [([1, 2, 3], 6), ([0, 0], 0), ([-1, 1], 0)]

    # 初始化环境
    env = CodeOptimizationEnv(initial_code, test_cases)
    env.seed(42)  # 设置环境种子

    # 加载训练好的模型
    model = PPO.load("ppo_code_optimizer")
    model.set_random_seed(42)  # 设置模型种子

    # 重置环境
    obs = env.reset()

    # 推理测试
    print("Initial Code:")
    print(env.current_code)
    print("\nOptimization Steps:")
    for i in range(10):
        action = model.predict(obs, deterministic=True)[0]  # 使用确定性预测
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}:")
        print(f"Code:\n{env.current_code}")
        print(f"Reward: {reward}")
        print("-" * 50)
        if done:
            print("Optimization stopped (max steps reached or low reward).")
            break
