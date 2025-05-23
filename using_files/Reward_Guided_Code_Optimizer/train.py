from env import CodeOptimizationEnv
from agent import CodeOptimizationAgent
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import random

if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer
    export CUDA_VISIBLE_DEVICES=2
    python train.py

    pip install stable-baselines3 gym shimmy astunparse
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    initial_code = """
def sum_list(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
"""
    test_cases = [([1, 2, 3], 6), ([0, 0], 0), ([-1, 1], 0)]
    env = make_vec_env(lambda: CodeOptimizationEnv(initial_code, test_cases), n_envs=1)
    env.seed(42)

    # 初始化并训练代理
    agent = CodeOptimizationAgent(env)
    best_reward = float("-inf")

    # 分段训练并保存最佳模型
    for _ in range(10000 // 2048):
        agent.train(total_timesteps=2048)
        obs = env.reset()
        total_reward = 0
        for _ in range(10):
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        if total_reward > best_reward:
            best_reward = total_reward
            agent.model.save("ppo_code_optimizer")
            print(f"Saved model with total reward: {total_reward}")

    # 测试最终模型
    env = CodeOptimizationEnv(initial_code, test_cases)
    obs = env.reset()
    print("Final Optimization Steps:")
    for i in range(10):
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f"Step {i+1}:")
        print(f"Code:\n{env.current_code}")
        print(f"Reward: {reward}")
        print("-" * 50)
        if done:
            print("Optimization stopped.")
            break
