from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from seo_environment import SEOEnvironment


def evaluate():

    env = make_vec_env(SEOEnvironment, n_envs=1)

    model = DQN.load("seo_rl_model")

    obs = env.reset()

    total_reward = 0

    for _ in range(50):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            break

    print("Total Optimization Reward:", total_reward)


def evaluate_multiple_episodes():

    env = make_vec_env(SEOEnvironment, n_envs=1)

    model = DQN.load("seo_rl_model")

    episode_rewards = []

    for episode in range(10):

        obs = env.reset()

        total_reward = 0

        for _ in range(50):

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward.item())

    print("Episode Rewards:", episode_rewards)


if __name__ == "__main__":
    evaluate()
    evaluate_multiple_episodes()