import gym
import ppaquette_gym_mallard_ducks
import tensorflow as tf

class Agent(object):
    """ The learning agent """

    def __init__(self, env_name, gamma=0.999, alpha=1e-4):
        """ Constructor """
        tf.reset_default_graph()
        self.gamma = gamma
        self.alpha = alpha

        # Inspecting env
        env = gym.make(env_name)
        self.nb_actions = env.action_space.n
        self.obs_space_shape = list(env.observation_space.shape)

        # Model
        self.placeholders = {'state': tf.placeholder(tf.float32, shape=[None] + self.obs_space_shape),
                             'action': tf.placeholder(tf.int32, shape=[None]),
                             'target': tf.placeholder(tf.float32, shape=[None]),
                             'old_value': tf.placeholder(tf.float32, shape=[None])}
        self.outputs = {}
        self.ops = {}
        self.build_model()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        """ Builds the model """
        state = self.placeholders['state']
        action = self.placeholders['action']
        target = self.placeholders['target']
        old_value = self.placeholders['old_value']
        delta = target - old_value

        # Policy network
        with tf.variable_scope('policy'):
            h1 = tf.layers.dense(state, 100, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
            logits = tf.layers.dense(h2, self.nb_actions, activation=None)
            probs = tf.nn.softmax(logits)
            log_probs = tf.log(probs)
            log_prob_chosen_action = -1. * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action)
            self.outputs['sampled_action'] = tf.multinomial(log_probs, num_samples=1)

        # Critic network
        with tf.variable_scope('critic'):
            v_h1 = tf.layers.dense(state, 100, activation=tf.nn.relu)
            value = tf.layers.dense(v_h1, 1, activation=None)
            self.outputs['value'] = value

        # Gradients
        with tf.variable_scope('gradients'):
            pg_loss = tf.reduce_sum(-delta * log_prob_chosen_action)
            mse_loss = tf.reduce_sum(tf.square(value - target))
            total_loss = pg_loss + 0.75 * mse_loss

            optimizer = tf.train.AdamOptimizer(self.alpha)
            minimize_op = optimizer.minimize(total_loss)

        # Ops
        self.ops['minimize_op'] = minimize_op

    def start_episode(self):
        """ Indicates to the agent that we are starting a new episode """
        pass

    def get_action(self, state):
        """ Returns the action to play in the given state """
        return min(self.nb_actions - 1,
                   self.session.run(self.outputs['sampled_action'], feed_dict={self.placeholders['state']: [state]})[0][0])

    def get_value(self, state):
        """ Returns the value of the given state """
        return self.session.run(self.outputs['value'], feed_dict={self.placeholders['state']: [state]})[0][0]

    def learn(self, transitions, nstep_returns):
        """ Learns from a transition """
        states, actions, rewards, next_states = zip(*transitions)
        old_values = self.session.run(self.outputs['value'], feed_dict={self.placeholders['state']: states})[:, 0]

        # Feed dict
        feed_dict = {self.placeholders['state']: states,
                     self.placeholders['action']: actions,
                     self.placeholders['target']: nstep_returns,
                     self.placeholders['old_value']: old_values}

        # Updating
        self.session.run(self.ops['minimize_op'], feed_dict=feed_dict)

    def compute_returns(self, transitions, nstep=5):
        """ Computes the N-Step Returns """
        returns = []
        nb_transitions = len(transitions)
        rewards = [reward for (state, action, reward, next_state) in transitions] + [0.] * nstep
        values = [agent.get_value(state) for (state, action, reward, next_state) in transitions] + [0.] * nstep

        # For each transition, in reverse order
        for transition_ix in reversed(range(nb_transitions)):
            current_return = values[transition_ix + nstep]
            for step_ix in reversed(range(nstep)):
                current_return = rewards[transition_ix + step_ix] + self.gamma * current_return
            returns += [current_return]

        # Returning n-step returns
        return list(reversed(returns))

def train_agent(agent, env_name, nb_episodes):
    """ Trains an agent on an env for a certain number of episodes """
    env = gym.make(env_name)
    rewards = []

    for episode_ix in range(nb_episodes):
        step = 0
        episode_reward = 0.
        agent.start_episode()
        done = False
        env_state = env.reset()
        transitions = []
        env_action = agent.get_action(env_state)

        # Generating an entire trajectory
        while not done and step <= 25:
            env_next_state, env_reward, done, _ = env.step(env_action)
            episode_reward += env_reward
            transitions += [(env_state, env_action, env_reward, env_next_state)]
            env_state = env_next_state
            step += 1

        # Computing N-Step Returns
        returns = agent.compute_returns(transitions)

        # Learning
        agent.learn(transitions, returns)

        # Recording episode rewards for plotting
        if (episode_ix + 1) % 100 == 0:
            print('.',)
            print('Total rew:', str(round(episode_reward, 2)),
                  ' -- Action:', env_action,
                  ' -- Birds killed/year:', str(round(0.3 * env_action, 2)))
        elif (episode_ix + 1) % 10 == 0:
            print('.', end='')
        rewards += [episode_reward]

if __name__ == '__main__':
    env_name = 'ppaquette/MallardDucks-v0'
    agent = Agent(env_name)
    train_agent(agent, env_name, nb_episodes=10000)
