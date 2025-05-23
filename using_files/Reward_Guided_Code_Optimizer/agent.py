from stable_baselines3 import PPO
from torch import nn


class CodeOptimizationAgent:
    def __init__(
        self,
        env,
        policy="MlpPolicy",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    ):
        self.model = PPO(
            policy=policy,
            env=env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            # policy_kwargs={
            #     "net_arch": [64, 64],
            #     "activation_fn": nn.ReLU,
            #     "ortho_init": True,
            # },
            verbose=1,
        )
        self.model.set_random_seed(42)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
