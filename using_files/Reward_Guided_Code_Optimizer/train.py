from env import CodeOptimizationEnv
from agent import CodeOptimizationAgent
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer
    export CUDA_VISIBLE_DEVICES=2
    python train.py

    pip install stable-baselines3 gym shimmy astunparse
    """
    initial_code = """
def sum_list(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
"""
    test_cases = [([1, 2, 3], 6), ([0, 0], 0), ([-1, 1], 0)]
    env = make_vec_env(lambda: CodeOptimizationEnv(initial_code, test_cases), n_envs=1)

    agent = CodeOptimizationAgent(env)
    agent.train(total_timesteps=10000)

    env = CodeOptimizationEnv(initial_code, test_cases)
    obs = env.reset()
    for _ in range(10):
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Code: {env.current_code}, Reward: {reward}")
        if done:
            break
