from math import sqrt, exp
import gym
from gym import utils, spaces
import numpy as np

EPSILON = 1e-10

class MallardDucksEnv(gym.Env, utils.EzPickle):
    metadata = {}

    def __init__(self):
        """ Constructor """
        utils.EzPickle.__init__(self)
        self.observation_space = spaces.Box(low=0., high=30., shape=(2,), dtype=np.float)   # (Nb birds, nb ponds)
        self.action_space = spaces.Discrete(100)                                            # Est nb of birds harvested

        # Current internal state
        self._nb_adult = 0.
        self._nb_ponds = 0.
        self._additive_mortality = True

    def step(self, action):
        """ Runs one timestep """
        adults_t = self._nb_adult
        ponds_t = self._nb_ponds
        nb_killed = action * 0.3
        nb_killed = np.random.choice([0.9 * nb_killed, nb_killed, 1.1 * nb_killed], 1, p=[0.2, 0.6, 0.2])[0]
        nb_killed = max(0, min(adults_t, nb_killed))

        # Computing the next state
        rain_t = np.random.normal(loc=16.46, scale=sqrt(4.41))
        young_t = (1. / (12.48 * (ponds_t + EPSILON) ** 0.851) + 0.519 / (adults_t + EPSILON)) ** -1.
        fall_t = 0.92 * adults_t + young_t
        ratio_killed_t = nb_killed / fall_t

        if self._additive_mortality:
            survival_rate_adult_t = max(0., min(1., 1. - 0.37 * exp(2.78 * ratio_killed_t)))
            survival_rate_young_t = max(0., min(1., 1. - 0.49 * exp(0.90 * ratio_killed_t)))
        elif ratio_killed_t >= 0.25:
            survival_rate_adult_t = max(0., min(1., 0.57 - 1.2 * (ratio_killed_t - 0.25)))
            survival_rate_young_t = max(0., min(1., 0.50 - 1.0 * (ratio_killed_t - 0.25)))
        else:
            survival_rate_adult_t = 0.57
            survival_rate_young_t = 0.50

        self._nb_adult = max(0., adults_t * survival_rate_adult_t + young_t * survival_rate_young_t)
        self._nb_ponds = max(0., -2.76 + 0.391 * ponds_t + 0.233 * rain_t)

        # Returning state, reward, done, info
        next_state = self._get_state()
        reward = nb_killed
        done = True if self._nb_adult < 0.01 else False
        info = {}
        return next_state, reward, done, info

    def reset(self, nb_adults=11., nb_ponds=1.5, additive_mortality=True):
        """ Resets the environment """
        self._nb_adult = nb_adults
        self._nb_ponds = nb_ponds
        self._additive_mortality = additive_mortality
        return self._get_state()

    def _get_state(self):
        """ Returns the current state """
        return self._nb_adult, self._nb_ponds
