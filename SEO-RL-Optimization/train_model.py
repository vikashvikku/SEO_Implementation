from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from seo_environment import SEOEnvironment


def train():

    env = make_vec_env(SEOEnvironment, n_envs=1)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=50000,
        batch_size=64,
        exploration_fraction=0.1,
        verbose=1
    )

    model.learn(total_timesteps=20000)

    model.save("seo_rl_model")

    print("Model training complete and saved.")


if __name__ == "__main__":
    train()