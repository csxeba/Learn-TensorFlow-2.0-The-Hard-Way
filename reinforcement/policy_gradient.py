import numpy as np
import gym
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl

from utility import plot_utils


def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def execute_episode(agent, environment, render=False):

    """
    Executes an episode from start to end and buffers the data required for learning.
    """

    _rewards = []
    _states = []
    _actions = []

    state = environment.reset()
    done = False
    reward = 0.
    step = 0

    while 1:

        if render:
            environment.render()

        _states.append(state)
        _rewards.append(reward)

        if done or step >= MAX_STEPS:
            break

        logits = agent(state[None, ...].astype("float32"))
        action = tf.random.categorical(logits, num_samples=1)[0].numpy()[0]
        _actions.append(action)

        state, reward, done, info = environment.step(action)
        step += 1

    return _states, _rewards, _actions


@tf.function(experimental_relax_shapes=True)  # Required, because input tensors have a variable batch size
def execute_learning_step(_states, _rewards, _actions):
    r_scaled = (_rewards - tf.reduce_mean(_rewards)) / tf.math.reduce_std(_rewards)

    with tf.GradientTape() as tape:

        logits = net(_states)

        # Policy gradient is <grad ( -log action prob. )>, so policy loss is <-log action prob.>,
        # which is cross entropy in case of discreete actions.
        loss = cross_entropy(_actions, logits, sample_weight=r_scaled)

    policy_gradient = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(policy_gradient, net.trainable_variables))

    return tf.reduce_mean(loss)


def simulate(agent, env, episodes, do_render=False, do_train=True):
    reward_history = []
    loss_history = []

    for episode in range(1, episodes + 1):

        states, rewards, actions = execute_episode(agent, env, do_render)
        reward_history.append(sum(rewards))

        if do_train:
            training_states = tf.convert_to_tensor(states[:-1], dtype=tf.float32)
            training_rewards = tf.convert_to_tensor(discount_rewards(rewards[1:]), dtype=tf.float32)
            training_actions = tf.convert_to_tensor(actions)
            training_loss = execute_learning_step(training_states, training_rewards, training_actions)
            loss_history.append(training_loss)

        print("\rEpisode {:>4} RWD {:>8.2f}".format(
            episode,
            np.mean(reward_history[-SMOOTHING_WINDOW_SIZE:])), end="")
        if do_train:
            print(" Loss {: >7.4f}  +- {:>7.4f}".format(
                np.mean(loss_history[-SMOOTHING_WINDOW_SIZE:]),
                np.std(loss_history[-SMOOTHING_WINDOW_SIZE:])), end="")

        if episode % SMOOTHING_WINDOW_SIZE == 0:
            print()
    print()
    return {"reward": np.array(reward_history), "loss": np.array(loss_history)}


EPISODES = 250
LEARNING_RATE = 1e-3
SMOOTHING_WINDOW_SIZE = 10
MAX_STEPS = 300

training_env = gym.make("CartPole-v1")
testing_env = gym.make("CartPole-v1")

net = tf.keras.models.Sequential([
    tfl.Dense(64, activation="relu"),
    tfl.Dense(64, activation="relu"),
    tfl.Dense(training_env.action_space.n)
])

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

training_history = simulate(net, training_env, episodes=EPISODES)

x = np.arange(len(training_history["reward"]))

fig, (top, bot) = plt.subplots(2, sharex="all", figsize=(16, 9))

plot_utils.plot_line_and_smoothing(x, training_history["reward"], smoothing_window=SMOOTHING_WINDOW_SIZE, axes_obj=top)
plot_utils.plot_line_and_smoothing(x, training_history["loss"], smoothing_window=SMOOTHING_WINDOW_SIZE, axes_obj=top)

top.set_title("Policy rewards")
bot.set_title("Policy losses")

plt.tight_layout()
plt.show()

simulate(net, testing_env, episodes=10, do_render=True, do_train=False)
