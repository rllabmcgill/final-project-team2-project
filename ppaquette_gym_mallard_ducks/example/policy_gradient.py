import gym
from matplotlib import pyplot as plt
import ppaquette_gym_mallard_ducks
import tensorflow as tf

class Agent(object):
    """ The learning agent """

    def __init__(self, env_name, gamma=0.95,
                 use_policy_trace=True, policy_alpha=0.03, policy_lambda=0.9,
                 use_critic_trace=True, critic_alpha=0.005, critic_lambda=0.9):
        """ Constructor """
        tf.reset_default_graph()
        self.gamma = gamma
        self.use_policy_trace = use_policy_trace
        self.policy_alpha = policy_alpha
        self.policy_lambda = policy_lambda
        self.use_critic_trace = use_critic_trace
        self.critic_alpha = critic_alpha
        self.critic_lambda = critic_lambda

        # Inspecting env
        env = gym.make(env_name)
        self.nb_actions = env.action_space.n
        self.obs_space_shape = list(env.observation_space.shape)

        # Model
        self.placeholders = {'state': tf.placeholder(tf.float32, shape=[1] + self.obs_space_shape),
                             'action': tf.placeholder(tf.int32, shape=[1]),
                             'target': tf.placeholder(tf.float32, shape=[1]),
                             'old_value': tf.placeholder(tf.float32, shape=[1])}
        self.outputs = {}
        self.ops = {}
        self.I = tf.get_variable('I', dtype=tf.float32, shape=(), trainable=False, initializer=tf.ones_initializer)
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
            logits = tf.layers.dense(state, self.nb_actions, activation=None)
            probs = tf.nn.softmax(logits)
            log_probs = tf.log(probs)
            log_prob_chosen_action = tf.gather(log_probs, action, axis=1)
            self.outputs['sampled_action'] = tf.multinomial(log_probs, num_samples=1)

        # Critic network
        with tf.variable_scope('critic'):
            value = tf.layers.dense(state, 1, activation=None)
            self.outputs['value'] = value

        # Gradients
        with tf.variable_scope('gradients'):
            policy_opt = tf.train.AdamOptimizer(self.policy_alpha)
            critic_opt = tf.train.AdamOptimizer(self.critic_alpha)
            policy_vars = tf.trainable_variables('policy')
            critic_vars = tf.trainable_variables('critic')
            policy_grads = tf.gradients(log_prob_chosen_action, policy_vars)
            critic_grads = tf.gradients(value, critic_vars)

        # Traces
        policy_traces = [1] * len(policy_vars)
        critic_traces = [1] * len(critic_vars)
        with tf.variable_scope('traces'):
            if self.use_policy_trace:
                policy_traces = [tf.get_variable(var.name.split(':')[0],
                                                 shape=var.shape,
                                                 dtype=var.dtype,
                                                 trainable=False,
                                                 initializer=tf.zeros_initializer) for var in tf.trainable_variables('policy')]
            if self.use_critic_trace:
                critic_traces = [tf.get_variable(var.name.split(':')[0],
                                                 shape=var.shape,
                                                 dtype=var.dtype,
                                                 trainable=False,
                                                 initializer=tf.zeros_initializer) for var in tf.trainable_variables('critic')]

        # Recalculating policy gradients
        policy_ops = []
        new_policy_grads = []
        for var, grad, trace in zip(*[policy_vars, policy_grads, policy_traces]):
            if self.use_policy_trace:
                updated_trace = self.gamma * self.policy_lambda * trace + self.I * grad
                policy_ops += [tf.assign(trace, updated_trace)]
                new_policy_grads += [-delta * updated_trace]
            else:
                new_policy_grads += [-delta * self.I * grad]
        policy_ops += [policy_opt.apply_gradients(zip(new_policy_grads, policy_vars))]

        # Recalculating critic gradients
        critic_ops = []
        new_critic_grads = []
        for var, grad, trace in zip(*[critic_vars, critic_grads, critic_traces]):
            if self.use_critic_trace:
                updated_trace = self.gamma * self.critic_lambda * trace + self.I * grad
                critic_ops += [tf.assign(trace, updated_trace)]
                new_critic_grads += [-delta * updated_trace]
            else:
                new_critic_grads += [-delta * self.I * grad]
        critic_ops += [critic_opt.apply_gradients(zip(new_critic_grads, critic_vars))]

        # Ops
        self.ops['reset_traces'] = tf.group(tf.variables_initializer([self.I]),
                                            tf.variables_initializer(policy_traces) if self.use_policy_trace else tf.no_op(),
                                            tf.variables_initializer(critic_traces) if self.use_critic_trace else tf.no_op())
        self.ops['critic_ops'] = tf.group(critic_ops)
        self.ops['policy_ops'] = tf.group(policy_ops)
        self.ops['decay_I'] = tf.assign(self.I, self.gamma * self.I)

    def start_episode(self):
        """ Resets the eligibility traces when starting a new episode """
        self.session.run(self.ops['reset_traces'])

    def get_action(self, state):
        """ Returns the action to play in the given state """
        return min(self.nb_actions - 1,
                   self.session.run(self.outputs['sampled_action'], feed_dict={self.placeholders['state']: [state]})[0][0])

    def get_value(self, state):
        """ Returns the value of the given state """
        return self.session.run(self.outputs['value'], feed_dict={self.placeholders['state']: [state]})[0][0]

    def learn(self, state, action, reward, next_state, is_done):
        """ Learns from a transition """
        state_value = self.get_value(state)
        next_state_value = self.get_value(next_state) if not is_done else 0.
        target = reward + self.gamma * next_state_value
        old_value = state_value

        # Feed dict
        feed_dict = {self.placeholders['state']: [state],
                     self.placeholders['action']: [action],
                     self.placeholders['target']: [target],
                     self.placeholders['old_value']: [old_value]}

        # Updating
        self.session.run(self.ops['critic_ops'], feed_dict=feed_dict)
        self.session.run(self.ops['policy_ops'], feed_dict=feed_dict)
        self.session.run(self.ops['decay_I'], feed_dict=feed_dict)

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

        while not done and step <= 100:
            env_action = agent.get_action(env_state)
            env_next_state, env_reward, done, _ = env.step(env_action)
            episode_reward += env_reward
            agent.learn(env_state, env_action, env_reward, env_next_state, done)
            env_state = env_next_state
            step += 1

        if (episode_ix + 1) % 100 == 0:
            print('.',)
        elif (episode_ix + 1) % 10 == 0:
            print('.', end='')
        rewards += [episode_reward]
        print(episode_reward)

    # Plotting rewards
    plt.plot(rewards)
    print(rewards)

env_name = 'ppaquette/MallardDucks-v0'
agent = Agent(env_name)
train_agent(agent, env_name, nb_episodes=1000)
