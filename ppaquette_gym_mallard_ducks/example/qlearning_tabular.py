""" Implements Tabular Q-Learning """
import gym
import ppaquette_gym_mallard_ducks
import numpy as np
np.set_printoptions(suppress=True, precision=3)

ALPHA = 0.05
GAMMA = 0.90

env = gym.make('ppaquette/MallardDucks-v0')
nb_actions = env.action_space.n
q_values = np.zeros((13, 7, nb_actions))        # 13x population size (6 - 18M), 7x ponds (0.5 - 3.5M),
q_values[:, :, :]= 5.                           # Exploring starts.
counts = np.zeros((13, 7, nb_actions))

def get_coordinates(population, nb_ponds):
    pop_distance = np.abs(np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) - population)
    pond_distance = np.abs(np.array([0.5, 1., 1.5, 2., 2.5, 3., 3.5]) - nb_ponds)
    return np.argmin(pop_distance), np.argmin(pond_distance)

def get_action(state, epsilon=0.2):
    """ Computes the action to take using epsilon-greedy """
    if np.random.random() <= epsilon:
        return np.random.randint(0, nb_actions)
    return np.random.choice(np.flatnonzero(q_values[get_coordinates(*state)] == q_values[get_coordinates(*state)].max()))

# Generating trajectories
trajectory_ix = 0
while np.min(counts) <= 100:
    trajectory_ix += 1
    starting_population = np.random.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    starting_ponds = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    env.unwrapped.configure(nb_adults=starting_population, nb_ponds=starting_ponds)
    state = env.reset()
    done = False
    step_ix = 0

    while not done:
        step_ix += 1
        action = get_action(state)
        next_state, reward, done, info = env.step(action)

        # Population is too low, or we didn't kill enough birds
        if next_state[0] < 5. or action/100. * state[0] <= 2.:
            reward -= 10.
            done = True

        (x1, y1), z1 = get_coordinates(*state), action
        x2, y2 = get_coordinates(*next_state)
        q_values[x1, y1, z1] = (1. - ALPHA) * q_values[x1, y1, z1] + ALPHA * (reward + GAMMA * np.max(q_values[x2, y2]))
        counts[x1, y1, z1] += 1

        state = next_state

    # Printing results
    if trajectory_ix % 10000 == 0:
        print()
        print('--- % to kill')
        print(np.argmax(q_values, axis=-1))

        print('\n--- # to kill')
        print(np.array([6,7,8,9,10,11,12,13,14,15,16,17,18])[:, None] * np.argmax(q_values, axis=-1)/100.)

        print('\n--- State values')
        print(np.max(q_values, axis=-1))

        print('\n--- Avg Min Visit', round(np.mean(np.min(counts, axis=-1)), 2), 'Min Visit', np.min(counts))

    elif (trajectory_ix + 1) % 1000 == 0:
        print('.', end='')
