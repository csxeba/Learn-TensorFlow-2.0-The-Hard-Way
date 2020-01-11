import numpy as np
import gym
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl

from utility import plot_utils


def _greedy(qs):
    return tf.argmax(qs, axis=-1)


def _epsilon(num_actions):
    return tf.random.uniform(1, minval=0, maxval=num_actions, dtype=tf.int64)


def epsilon_greedy(qs):
    global epsilon
    num_actions = qs.shape[0]
    roll = np.random.random()
    if roll < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(qs)
    epsilon *= epsilon_decay
    epsilon = max(epsilon, epsilon_min)
    return action


@tf.function
def learn_step(inputs: tf.Tensor, targets: tf.Tensor):
    with tf.GradientTape() as tape:
        qs = net(inputs)
        loss = loss_function(targets, qs)

    gradient = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradient, net.trainable_variables))

    return tf.reduce_mean(loss)


class LearningBuffer:

    def __init__(self, state_shape: tuple):
        self.states = np.empty((BUFFER_SIZE,) + state_shape, dtype="float32")
        self.states_next = np.empty((BUFFER_SIZE,) + state_shape, dtype="float32")
        self.actions = np.empty(BUFFER_SIZE, dtype="int32")
        self.rewards = np.empty(BUFFER_SIZE, dtype="float32")
        self.dones = np.empty(BUFFER_SIZE, dtype=bool)
        self.pointer = 0
        self.full = False

    def push(self, state, state_next, action, reward, done):
        self.states[self.pointer] = state
        self.states_next[self.pointer] = state_next
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done

        self.pointer += 1
        if self.pointer == BUFFER_SIZE:
            self.full = True
        self.pointer %= BUFFER_SIZE

    def sample(self, batch_size):
        if not self.full:
            raise RuntimeError("Completely fill the buffer before sampling!")
        arg = np.random.randint(0, BUFFER_SIZE, size=batch_size)
        return self.states[arg], self.states_next[arg], self.actions[arg], self.rewards[arg], self.dones[arg]

    def make_batch(self, batch_size, agent):

        S, S_, A, R, D = self.sample(batch_size)
        target = agent(S).numpy()
        Q_target = tf.reduce_max(agent(S_), axis=1).numpy()
        bellman_reserve = GAMMA * Q_target + R

        target[range(len(S)), A] = bellman_reserve
        target[D, A[D]] = R[D]

        return tf.convert_to_tensor(S), tf.convert_to_tensor(target)


def simulate(agent: tf.keras.Model, env, buffer: LearningBuffer, episodes, learning_batch_size=None, verbose=1):

    reward_history = []
    loss_history = []
    q_histsory = []

    for episode in range(1, episodes + 1):

        state = training_env.reset()
        done = False
        reward_sum = 0.
        step = 0
        losses = 0.
        Qs = 0.

        while not done:
            Q = agent(state[None, ...])[0]
            Qs += tf.reduce_max(Q).numpy()
            action = epsilon_greedy(Q)

            next_state, reward, done, info = env.step(action)

            buffer.push(state, next_state, action, reward, done)

            if learning_batch_size:
                inputs, targets = buffer.make_batch(learning_batch_size, agent)
                loss = learn_step(inputs, targets)
                losses += loss

            reward_sum += reward
            step += 1
            state = next_state

            if step >= MAX_STEP:
                break

        reward_history.append(reward_sum)
        loss_history.append(losses / (step+1))
        q_histsory.append(Qs / (step+1))

        if verbose:
            print(f"\rEpisode {episode:>5} -"
                  f" R {np.mean(reward_history[-SMOOTHING_WINDOW_SIZE:]):>6.1f}"
                  f" Q {np.mean(q_histsory[-SMOOTHING_WINDOW_SIZE:]):>7.2f}"
                  f" Loss {np.mean(loss_history[-SMOOTHING_WINDOW_SIZE:]):>7.4f}"
                  f" +- {np.std(loss_history[-SMOOTHING_WINDOW_SIZE:]):>7.4f}"
                  f" Eps {epsilon:>6.2%}", end="")

            if episode % SMOOTHING_WINDOW_SIZE == 0:
                print()

    return {"loss": loss_history, "reward": reward_history}


EPISODES = 300
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SMOOTHING_WINDOW_SIZE = 10
GAMMA = 0.99
BUFFER_SIZE = 1000
MAX_STEP = 200

training_env = gym.make("CartPole-v1")

num_actions = training_env.action_space.n

net = tf.keras.models.Sequential([
    tfl.Dense(400, activation="relu"),
    tfl.Dense(300, activation="relu"),
    tfl.Dense(num_actions)
])

loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

memory_buffer = LearningBuffer(state_shape=training_env.observation_space.shape)

epsilon = 1.
epsilon_decay = 1.
epsilon_min = 0.1

while not memory_buffer.full:
    simulate(net, training_env, memory_buffer, episodes=1, learning_batch_size=0, verbose=0)
    print(f"\rFilling replay memory... {memory_buffer.pointer / BUFFER_SIZE:>7.2%}", end="")
print(f"\rFilling replay memory... 100.00%")

epsilon = 0.1
epsilon_decay = 1.

training_history = simulate(net, training_env, memory_buffer, EPISODES, BATCH_SIZE, verbose=1)

reward_buffer, loss_buffer = training_history["reward"], training_history["loss"]

x = np.arange(len(reward_buffer))

fig, (top, bot) = plt.subplots(2, sharex="all", figsize=(16, 9))

plot_utils.plot_line_and_smoothing(x, np.array(reward_buffer), smoothing_window=SMOOTHING_WINDOW_SIZE, axes_obj=top)
plot_utils.plot_line_and_smoothing(x, np.array(loss_buffer), smoothing_window=SMOOTHING_WINDOW_SIZE, axes_obj=top)

top.set_title("DQN rewards")
bot.set_title("DQN losses")

plt.tight_layout()
plt.show()
