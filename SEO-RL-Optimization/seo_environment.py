import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SEOEnvironment(gym.Env):
    """
    AI-Driven Reinforcement Learning Environment
    for SEO and Voice Search Optimization
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # State:
        # [SERP rank, CTR, Bounce Rate, Dwell Time, Voice Query Score]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Actions
        # 0: Optimize title
        # 1: Add FAQ (voice optimization)
        # 2: Improve content depth
        # 3: Add schema markup
        self.action_space = spaces.Discrete(4)

        self.state = None
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.random.rand(5).astype(np.float32)
        self.steps = 0

        return self.state, {}

    def step(self, action):

        serp, ctr, bounce, dwell, voice = self.state

        # Apply action effects
        if action == 0:      # Title optimization
            ctr += 0.05

        elif action == 1:    # FAQ (voice search optimization)
            voice += 0.10
            ctr += 0.03

        elif action == 2:    # Content improvement
            dwell += 0.10
            bounce -= 0.05

        elif action == 3:    # Schema markup
            ctr += 0.04

        # Clamp values between 0 and 1
        ctr = np.clip(ctr, 0, 1)
        bounce = np.clip(bounce, 0, 1)
        dwell = np.clip(dwell, 0, 1)
        voice = np.clip(voice, 0, 1)

        # Reward function
        reward = (2 * ctr) + dwell - bounce

        # SERP improves when reward increases
        serp = np.clip(serp - reward * 0.05, 0, 1)

        self.state = np.array([serp, ctr, bounce, dwell, voice], dtype=np.float32)

        self.steps += 1

        terminated = self.steps >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}