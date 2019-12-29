import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from utility import data_utils, eval_utils


@tf.function
def sample(z_mean, z_log_var):
    noise = tf.keras.backend.random_normal(z_mean.shape)
    z_std = tf.sqrt(tf.exp(z_log_var))
    z = noise * z_std + z_mean
    return z


@tf.function
def kld(z_mean, z_log_var):
    z = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl = -0.5 * tf.reduce_sum(z, axis=1)
    return kl


@tf.function
def sse(y_true, y_pred):
    d = tf.reduce_sum(tf.square(y_true - y_pred), axis=(1, 2, 3))
    return d


class VAE(tf.keras.Model):

    class Encoder(tf.keras.Model):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.inputs = tfl.Input((28, 28, 1))
            self.flat = tfl.Flatten()
            self.d1 = tfl.Dense(64, activation="relu")  # 12
            self.d2 = tfl.Dense(32, activation="relu")  # 4
            self.z_mean = tfl.Dense(32)
            self.z_std = tfl.Dense(32)

        def call(self, x, training=None, mask=None):
            x = self.flat(x)
            x = self.d1(x)
            x = self.d2(x)
            z_mean = self.z_mean(x)
            z_std = self.z_std(x)
            return z_mean, z_std

    class Decoder(tf.keras.models.Sequential):

        def __init__(self):
            layers = [
                tfl.Dense(32, input_dim=32, activation="relu"),
                tfl.Dense(32, activation="relu"),
                tfl.Dense(64, activation="relu"),
                tfl.Dense(28*28),
                tfl.Reshape((28, 28, 1))
            ]

            super().__init__(layers)

    def __init__(self,
                 loss: tf.keras.losses.Loss = "default",
                 optimizer: tf.keras.optimizers.Optimizer = "default"):

        super().__init__()
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        if loss == "default":
            loss = sse
        if optimizer == "default":
            optimizer = tf.keras.optimizers.Adam()
        self.loss = loss
        self.optimizer = optimizer
        self.loss_reduce = tf.keras.metrics.Mean("loss")
        self.kld_reduce = tf.keras.metrics.Mean("kld")

    @tf.function
    def encode(self, x):
        z_mean, z_std = self.encoder(x)
        return z_mean

    @tf.function
    def sample(self, x):
        z_mean, z_std = self.encoder(x)
        z = sample(z_mean, z_std)
        return z

    @tf.function
    def reconstruct(self, x):
        z_mean, z_log_var = self.encoder(x)
        reconstruction = self.decoder(z_mean)
        return reconstruction

    @tf.function
    def decode(self, z):
        image = self.decoder(z)
        return image

    @tf.function()
    def generate(self, n):
        zs = tf.random.normal((n, 32))
        return self.decode(zs)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var = self.encoder(inputs)
        z = sample(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction

    def _train_step(self, x):

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = sample(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            loss = self.loss(x, reconstruction)
            kldiv = kld(z_mean, z_log_var)
            loss = loss + kldiv

        gradient = tape.gradient(loss, self.encoder.weights + self.decoder.weights)
        self.optimizer.apply_gradients(zip(gradient, self.encoder.weights + self.decoder.weights))

        self.loss_reduce(loss)
        self.kld_reduce(kldiv)

    def train_step(self, x):
        self._train_step(x)
        loss = self.loss_reduce.result()
        kldiv = self.kld_reduce.result()
        self.loss_reduce.reset_states()
        self.kld_reduce.reset_states()
        return loss, kldiv

    def epoch(self, ds, steps_per_epoch):
        losses = []
        klds = []
        for i, x in enumerate(ds, start=1):
            loss, kldiv = self.train_step(x)
            losses.append(loss)
            klds.append(kldiv)
            print(f"\rP {i / steps_per_epoch:>7.2%}"
                  f" - loss {np.mean(losses[-100:]):7.4f}"
                  f" - kld {np.mean(klds[-100:]):.4f}",
                  end="")
            if i >= steps_per_epoch:
                break
        print()
        return {"losses": losses, "kldivs": klds}


def execute_training():
    optimizer = tf.keras.optimizers.Adam(0.001)

    vae = VAE(optimizer=optimizer)
    data = data_utils.MNIST(batch_size=32)
    dataset = data.train_dataset(include_labels=False)

    for epoch in range(1, 31):
        print("Epoch", epoch)
        vae.epoch(dataset, steps_per_epoch=data.train_steps_per_epoch)
        if epoch % 10 == 0:
            print("Dumping montage...")
            eval_utils.create_montage(vae, "vae_montage_epoch_{}.png".format(epoch))


if __name__ == '__main__':
    execute_training()
