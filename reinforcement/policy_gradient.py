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


def execute_episode(environment, agent):

    """
    Executes an episode from start to end and buffers the data required for learning.
    """

    _rewards = []
    _states = []
    _actions = []

    state = environment.reset()
    done = False
    reward = 0.

    while 1:

        _states.append(state)
        _rewards.append(reward)

        if done:
            break

        logits = agent(state[None, ...])
        action = tf.random.categorical(logits, num_samples=1)[0].numpy()[0]
        _actions.append(action)

        state, reward, done, info = environment.step(action)

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


EPISODES = 500
LEARNING_RATE = 1e-3
SMOOTHING_WINDOW_SIZE = 10

training_env = gym.make("CartPole-v1")

net = tf.keras.models.Sequential([
    tfl.Dense(64, activation="relu"),
    tfl.Dense(64, activation="relu"),
    tfl.Dense(2)
])

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="policy_loss")
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

reward_buffer = []
loss_buffer = []

for episode in range(1, EPISODES+1):

    states, rewards, actions = execute_episode(training_env, net)

    training_states = tf.convert_to_tensor(states[:-1])
    training_rewards = tf.convert_to_tensor(discount_rewards(rewards[1:]))
    training_actions = tf.convert_to_tensor(actions)

    training_loss = execute_learning_step(training_states, training_rewards, training_actions)

    reward_buffer.append(sum(rewards))
    loss_buffer.append(training_loss)

    print("\rEpisode {:>4} RWD {:>8.2f} Loss {: >7.4f}  +- {:>7.4f}".format(
        episode,
        np.mean(reward_buffer[-SMOOTHING_WINDOW_SIZE:]),
        np.mean(loss_buffer[-SMOOTHING_WINDOW_SIZE:]),
        np.std(loss_buffer[-SMOOTHING_WINDOW_SIZE:])), end="")
    if episode % SMOOTHING_WINDOW_SIZE == 0:
        print()

print()

x = np.arange(len(reward_buffer))

fig, (top, bot) = plt.subplots(2, sharex="all", figsize=(16, 9))

plot_utils.plot_line_and_smoothing(x, np.array(reward_buffer), smoothing_window=SMOOTHING_WINDOW_SIZE, axes_obj=top)
plot_utils.plot_line_and_smoothing(x, np.array(loss_buffer), smoothing_window=SMOOTHING_WINDOW_SIZE, axes_obj=top)

top.set_title("Policy rewards")
bot.set_title("Policy losses")

plt.tight_layout()
plt.show()
