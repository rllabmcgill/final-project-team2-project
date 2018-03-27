from gym.envs.registration import register
from .package_info import USERNAME
from .mallard_ducks_env import MallardDucksEnv


# Env registration
# ==========================
register(
    id='{}/MallardDucks-v0'.format(USERNAME),
    entry_point='{}_gym_mallard_ducks:MallardDucksEnv'.format(USERNAME),
    max_episode_steps=100,
    nondeterministic=True,
)
