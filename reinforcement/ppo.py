import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.optimizers as opt

from utility import plot_utils, history, rl_utils


class PPO:

    logging = ["actor_loss", "actor_loss_std", "value", "advantage", "adv_std", "entropy", "kl_div",
               "critic_loss"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 actor_optimizer: tf.keras.optimizers.Optimizer,
                 critic_optimizer: tf.keras.optimizers.Optimizer,
                 discount_factor_gamma=0.99,
                 ratio_clip=0.1,
                 entropy_penalty_beta=0.01):

        self.actor = actor
        self.critic = critic
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.gamma = discount_factor_gamma
        self.beta = entropy_penalty_beta
        self.clip = ratio_clip
        self.num_actions = self.actor.output_shape[-1]

    @tf.function
    def train_critic(self, state, state_next, reward, done):
        state = tf.cast(state, tf.float32)
        state_next = tf.cast(state_next, tf.float32)
        value_next = self.critic(state_next)
        bellman_reserve = value_next * self.gamma * (1 - tf.cast(done, tf.float32)) + reward
        with tf.GradientTape() as tape:
            value = self.critic(state)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(bellman_reserve, value))
        gradients = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(gradients, self.critic.trainable_weights))
        return loss, tf.reduce_mean(value)

    @tf.function
    def train_actor(self, states, actions, rewards, old_probabilities):
        states = tf.cast(states, tf.float32)
        values = self.critic(states)
        advantages = rewards - values
        action_onehot = tf.one_hot(actions, depth=self.num_actions)

        adv_mean = tf.reduce_mean(advantages)
        adv_std = tf.math.reduce_std(advantages)
        advantages = advantages - adv_mean
        advantages = advantages / adv_std

        old_probabilities_masked = tf.reduce_sum(action_onehot * old_probabilities, axis=1)
        old_log_prob = tf.math.log(old_probabilities_masked)

        with tf.GradientTape() as tape:
            new_pedictions = self.actor(states)
            new_probabilities = tf.reduce_sum(action_onehot * tf.keras.activations.softmax(new_pedictions), axis=1)
            new_log_prob = tf.math.log(new_probabilities)
            entropy = -tf.reduce_mean(new_log_prob)

            selection = tf.cast(advantages > 0, tf.float32)
            min_adv = (1+self.clip) * advantages * selection + (1-self.clip) * advantages * (1-selection)
            ratio = tf.exp(new_log_prob - old_log_prob)
            utilities = -tf.minimum(ratio*advantages, min_adv)
            utility = tf.reduce_mean(utilities)
            loss = -entropy * self.beta + utility

        gradients = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(gradients, self.actor.trainable_weights))

        kld = tf.reduce_mean(old_probabilities_masked * (old_log_prob - new_log_prob))
        utility_stdev = tf.math.reduce_std(utilities)

        return utility, utility_stdev, adv_mean, adv_std, entropy, kld

    def train_step(self, state, state_next, action, reward, done, old_probability):
        critic_loss, value = self.train_critic(state, state_next, reward, done)
        actor_loss, actor_loss_std, advantage, adv_std, entropy, kl_div = self.train_actor(
            state, action, reward, old_probability)
        vars = locals()
        return {k: v for k, v in locals().items() if k in self.logging}

    def fit(self, epochs, batch_size, state, state_next, action, reward, done, old_probability):
        discounted_reward = rl_utils.discount_rewards(reward, done, self.gamma).astype("float32")
        no_samples = len(state)
        datasets = tuple(map(
            tf.data.Dataset.from_tensor_slices, [state, state_next, action, discounted_reward, done, old_probability]))
        ds = tf.data.Dataset.zip(datasets).shuffle(buffer_size=no_samples).batch(batch_size)
        train_history = history.History(*(["R"] + self.logging))
        for epoch in range(1, epochs+1):
            for data in ds:
                logs = self.train_step(*data)
                train_history.record(**logs)
        return train_history


def execute_episode(agent, environment, render=False):
    states = []
    actions = []
    rewards = []
    dones = []
    old_probabilities = []

    state = environment.reset()
    done = False
    reward = 0.
    step = 0

    while 1:

        if render:
            environment.render()

        states.append(state)
        rewards.append(reward)
        dones.append(done)

        if done or step >= MAX_STEPS:
            break

        logits = agent.actor(state[None, ...].astype("float32"))
        probabilities = tf.keras.activations.softmax(logits).numpy()[0]
        if np.any(np.isnan(probabilities)):
            raise RuntimeError()
        action = tf.random.categorical(logits, num_samples=1)[0].numpy()[0]
        actions.append(action)
        old_probabilities.append(probabilities)

        state, reward, done, info = environment.step(int(action))
        step += 1

    return states, actions, rewards, dones, old_probabilities


def simulate(agent: PPO, env, episodes: int, do_render=False, do_train=True):
    simulation_history = history.History(*(["R"] + PPO.logging))

    simulation_history.print_header()
    for episode in range(1, episodes + 1):

        states, actions, rewards, dones, old_probabilities = execute_episode(agent, env, do_render)
        simulation_history.record(R=sum(rewards[1:]))

        if do_train:
            training_histroy = agent.fit(
                EPOCHS, BATCH_SIZE, states[:-1], states[1:], actions, rewards[1:], dones[1:], old_probabilities)
            simulation_history.incorporate(training_histroy, do_reduction=True)

        simulation_history.print(average_last=SMOOTHING_WINDOW_SIZE, return_carriege=True)

        if episode % SMOOTHING_WINDOW_SIZE == 0:
            print()
            if episode % (SMOOTHING_WINDOW_SIZE*10) == 0:
                print()
                simulation_history.print_header()
    print()
    return simulation_history


def experiment():
    env = gym.make("CartPole-v1")
    input_shape = env.observation_space.shape
    output_dim = env.action_space.n

    actor = tf.keras.Sequential([
        tfl.Dense(64, input_shape=input_shape, activation="relu"),
        tfl.Dense(64, activation="relu"),
        tfl.Dense(output_dim)
    ])
    critic = tf.keras.Sequential([
        tfl.Dense(64, input_shape=input_shape, activation="relu"),
        tfl.Dense(64, activation="relu"),
        tfl.Dense(1)
    ])
    agent = PPO(actor, critic,
                actor_optimizer=opt.Adam(ACTOR_LR),
                critic_optimizer=opt.Adam(CRITIC_LR),
                discount_factor_gamma=DISCOUNT_FACTOR_GAMMA,
                ratio_clip=RATIO_CLIP,
                entropy_penalty_beta=ENTROPY_PENALTY_BETA)
    simulation_history = simulate(agent, env, episodes=EPISODES, do_render=False, do_train=True)
    plot_utils.plot_history(simulation_history, SMOOTHING_WINDOW_SIZE, skip_first=10, show=True, figsize=(16, 9))


MAX_STEPS = 200
SMOOTHING_WINDOW_SIZE = 10
EPISODES = 300
EPOCHS = 10
BATCH_SIZE = 32
DISCOUNT_FACTOR_GAMMA = 0.99
ENTROPY_PENALTY_BETA = 0.05
RATIO_CLIP = 0.2
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4

experiment()
