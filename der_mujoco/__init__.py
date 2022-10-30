from gym.envs.registration import (
    register,
)

register(
    id="DerRope1D-v0",
    entry_point="der_mujoco.envs:DerRope1DEnv",
    max_episode_steps=10000,
)