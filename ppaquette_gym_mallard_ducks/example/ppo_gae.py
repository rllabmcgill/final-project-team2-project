import gym
import ppaquette_gym_mallard_ducks
import tensorflow as tf

class Agent(object):
    """ The learning agent """

    def __init__(self, env_name, gamma=0.999, lambda_=0.3, alpha=1e-4, epsilon=0.5):
        """ Constructor """
        tf.reset_default_graph()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epsilon = epsilon

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
        # New Policy (Current Weights)
        with tf.variable_scope('new_policy'):
            new_h1 = tf.layers.dense(state, 100, activation=tf.nn.relu)
            new_h2 = tf.layers.dense(new_h1, 100, activation=tf.nn.relu)
            new_logits = tf.layers.dense(new_h2, self.nb_actions, activation=None)
            new_probs = tf.nn.softmax(new_logits)
            new_log_probs = tf.log(new_probs)
            new_log_prob_chosen_action = -1. * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new_logits, labels=action)
            self.outputs['sampled_action'] = tf.multinomial(new_log_probs, num_samples=1)

        # Old Policy (Previous weights)
        with tf.variable_scope('old_policy'):
            old_h1 = tf.layers.dense(state, 100, activation=tf.nn.relu)
            old_h2 = tf.layers.dense(old_h1, 100, activation=tf.nn.relu)
            old_logits = tf.layers.dense(old_h2, self.nb_actions, activation=None)
            old_log_prob_chosen_action = -1. * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=old_logits, labels=action)

        # Critic network
        with tf.variable_scope('critic'):
            v_h1 = tf.layers.dense(state, 100, activation=tf.nn.relu)
            value = tf.layers.dense(v_h1, 1, activation=None)
            self.outputs['value'] = value

        # Computing entropy
        log_p_num = new_logits - tf.reduce_max(new_logits, axis=-1, keepdims=True)
        p_num = tf.exp(log_p_num)
        p_den = tf.reduce_sum(p_num, axis=-1, keepdims=True)
        entropy = tf.reduce_sum(p_num / p_den * (tf.log(p_den) - log_p_num), axis=-1)

        # Gradients
        with tf.variable_scope('gradients'):
            ratio = tf.exp(new_log_prob_chosen_action - old_log_prob_chosen_action)
            clipped_ratio = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon)

            surrogate_loss_1 = ratio * delta
            surrogate_loss_2 = clipped_ratio * delta
            surrogate_loss = -tf.reduce_mean(tf.minimum(surrogate_loss_1, surrogate_loss_2))

            mse_loss = tf.reduce_sum(tf.square(value - target))
            entropy_loss = -tf.reduce_mean(entropy)
            total_loss = surrogate_loss + 0.75 * mse_loss + 0.05 * entropy_loss

            train_vars = tf.trainable_variables(scope='new_policy') + tf.trainable_variables(scope='critic')
            optimizer = tf.train.AdamOptimizer(self.alpha)
            minimize_op = optimizer.minimize(total_loss, var_list=train_vars)

        # Assign New to Old
        new_policy_vars = tf.trainable_variables(scope='new_policy')
        old_policy_vars = tf.trainable_variables(scope='old_policy')
        assign_new_to_old_op = tf.group(*[tf.assign(old_var, new_var) for new_var, old_var in zip(*[new_policy_vars, old_policy_vars])])

        # Ops
        self.ops['minimize_op'] = minimize_op
        self.ops['assign_op'] = assign_new_to_old_op

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

        self.session.run(self.ops['minimize_op'], feed_dict=feed_dict)
        self.session.run(self.ops['assign_op'])

    def compute_returns(self, transitions):
        """ Computes the GAE Returns """
        returns = []
        nb_transitions = len(transitions)
        values = [agent.get_value(state) for (state, action, reward, next_state) in transitions] + [0.]
        last_gae = 0.

        # For each transition, in reverse order
        for transition_ix in reversed(range(nb_transitions)):
            state, action, reward, next_state = transitions[transition_ix]
            current_value = values[transition_ix]
            next_value = values[transition_ix + 1]

            delta = reward + self.gamma * next_value - current_value
            advantage = delta + self.gamma * self.lambda_ * last_gae
            returns += [advantage + current_value]
            last_gae = advantage

        # Returning targets
        return list(reversed(returns))

def train_agent(agent, env_name, nb_episodes):
    """ Trains an agent on an env for a certain number of episodes """
    env = gym.make(env_name)
    rewards = []

    for episode_ix in range(nb_episodes):
        batch_transitions = []
        batch_returns = []

        for _ in range(3):
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

            # Computing Returns
            returns = agent.compute_returns(transitions)

            # Adding to batch
            batch_transitions += transitions
            batch_returns += returns

        # Learning
        agent.learn(batch_transitions, batch_returns)

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
