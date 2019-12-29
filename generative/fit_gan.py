import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl

from utility import data_utils, eval_utils


class GAN:

    """
    GAN setup is as follows:

    [Database of images]-+
                         +----[Discriminator]-(0-1 Fake or Real?)-(Cross entropy loss)
    [Generator]----------+

    """

    class Generator(tf.keras.Sequential):

        def __init__(self, adam_lr=1e-3):
            layers = [
                tfl.Dense(32, input_dim=32),
                tfl.BatchNormalization(),
                tfl.LeakyReLU(),
                tfl.Dense(64),
                tfl.BatchNormalization(),
                tfl.LeakyReLU(),
                tfl.Dense(28*28),
                tfl.Reshape((28, 28, 1)),
            ]
            super().__init__(layers)
            self.optimizer = tf.keras.optimizers.Adam(adam_lr)

        @tf.function
        def step(self, m, loss_fn, discriminator: "GAN.Discriminator"):
            z = tf.random.normal([m, 32])
            label = tf.ones(m)
            with tf.GradientTape() as tape:
                fake = self(z)
                fakeness = discriminator(fake)
                loss = loss_fn(label, fakeness)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return tf.reduce_mean(loss)

    class Discriminator(tf.keras.Sequential):

        def __init__(self, adam_lr=1e-3):
            layers = [
                tfl.Flatten(input_shape=(28, 28, 1)),
                tfl.Dense(64),
                tfl.Dropout(0.5),
                tfl.BatchNormalization(),
                tfl.LeakyReLU(),
                tfl.Dense(32),
                tfl.BatchNormalization(),
                tfl.Activation("tanh"),
                tfl.Dense(1)
            ]
            super().__init__(layers)
            self.optimizer = tf.keras.optimizers.Adam(adam_lr)

        @tf.function
        def step(self, x, loss_fn, generator: "GAN.Generator"):
            m = x.shape[0]
            z = tf.random.normal([m, 32])

            fakes = generator(z)

            inputs = tf.concat([x, fakes], axis=0)
            y = tf.concat([tf.ones(m), tf.zeros(m)], axis=0)

            with tf.GradientTape(persistent=True) as tape:
                fakeness = self(inputs)
                loss = loss_fn(y, fakeness)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return tf.reduce_mean(loss)

    def __init__(self, generator_adam_lr, discriminator_adam_lr):
        self.generator = self.Generator(generator_adam_lr)
        self.discriminator = self.Discriminator(discriminator_adam_lr)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def generate(self, z=None, n=1):
        if z is None:
            z = tf.random.normal([n, 32])
        image = self.generator(z)
        return image

    @tf.function
    def evaluate(self, x):
        m = x.shape[0]
        fakes = self.generate(n=m)
        fakeness = tf.sigmoid(self.discriminator(tf.concat([x, fakes], axis=0)))
        y = tf.concat([tf.ones(m), tf.zeros(m)], axis=0)[..., None]
        acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y, fakeness))
        return acc

    def epoch(self, ds, steps_per_epoch):
        d_losses = []
        g_losses = []
        accs = []
        for i, x in enumerate(ds, start=1):
            acc = self.evaluate(x)
            d_loss = self.discriminator.step(x, self.loss_fn, self.generator)
            g_loss = self.generator.step(x.shape[0], self.loss_fn, self.discriminator)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
            accs.append(acc.numpy())
            print(
                f"\rP {i / steps_per_epoch:>7.2%} - D {np.mean(d_losses[-100:]):>7.4f} "
                f"G {np.mean(g_losses[-100:]):>7.4f} Acc {np.mean(accs[-100:]):>7.2%}",
                end="")
            if i >= steps_per_epoch:
                break

        print()


def execute_training():
    data = data_utils.MNIST(batch_size=32)
    dataset = data.train_dataset(include_labels=False)

    gan = GAN(generator_adam_lr=2e-4, discriminator_adam_lr=1e-4)

    for epoch in range(1, 31):
        print("Epoch", epoch)
        gan.epoch(dataset, steps_per_epoch=data.train_steps_per_epoch)
        print()
        if epoch % 10 == 0:
            print("Dumping montage...")
            eval_utils.create_montage(gan, "gan_montage_epoch_{}.png".format(epoch))

    for i in range(100):
        image = gan.generate()[0, ..., 0]
        image = np.clip(image * 255, 0, 255).astype("uint8")
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    execute_training()
