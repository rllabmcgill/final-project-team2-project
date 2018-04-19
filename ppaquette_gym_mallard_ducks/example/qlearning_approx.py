""" Implements Tabular Q-Learning """
import gym
import ppaquette_gym_mallard_ducks
import numpy as np
import tensorflow as tf
np.set_printoptions(suppress=True, precision=3)

ALPHA = 0.05
GAMMA = 0.90

env = gym.make('ppaquette/MallardDucks-v0')
nb_actions = env.action_space.n

# Building network
tf_state = tf.placeholder(tf.float32, shape=[None, 2])
tf_targets = tf.placeholder(tf.float32, shape=[None])
h1 = tf.layers.dense(tf_state, nb_actions, activation=tf.nn.relu)
q_values = tf.layers.dense(h1, nb_actions, activation=None)
l2_loss = tf.reduce_sum(tf.square(tf_targets - q_values))
optimizer_op = tf.train.AdamOptimizer().minimize(l2_loss)
session = tf.Session()
session.run(tf.global_variables_initializer())

def get_action(state, epsilon=0.1):
    if np.random.random() <= epsilon:
        return np.random.randint(0, nb_actions)
    return np.argmax(session.run(q_values, feed_dict={tf_state: [state]})[0])

# Generating trajectories
nb_trajectories = 1000000
for trajectory_ix in range(nb_trajectories):
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

        next_q_values = session.run(q_values, feed_dict={tf_state: [next_state]})[0]
        targets = reward + GAMMA * np.max(next_q_values)

        # Updating
        session.run(optimizer_op, feed_dict={tf_state: [state], tf_targets: [targets]})
        state = next_state

    # Printing results
    if (trajectory_ix + 1) % 2000 == 0:
        perc_to_kill = np.zeros((13, 7), dtype=np.float32)
        nb_to_kill = np.zeros((13, 7), dtype=np.float32)
        state_values = np.zeros((13, 7), dtype=np.float32)

        # Looping over each pond size
        for pond_ix, pond_size in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
            states = [[population, pond_size] for population in range(6, 19)]
            pond_q_values = session.run(q_values, feed_dict={tf_state: states})
            perc_to_kill[:, pond_ix] = np.argmax(pond_q_values, axis=-1)
            nb_to_kill[:, pond_ix] = np.argmax(pond_q_values, axis=-1) / 100. * np.array(range(6, 19))
            state_values[:, pond_ix] = np.max(pond_q_values, axis=-1)

        print()
        print('--- % to kill')
        print(perc_to_kill)

        print('\n--- # to kill')
        print(nb_to_kill)

        print('\n--- State values')
        print(state_values)

    elif (trajectory_ix + 1) % 1000 == 0:
        print('.', end='')
