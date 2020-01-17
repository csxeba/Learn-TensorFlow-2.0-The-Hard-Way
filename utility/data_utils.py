import math

import tensorflow as tf


def _make_dataset(feed, batch_size, include_labels):
    if not include_labels:
        feed = feed[0]
    ds = (tf.data.Dataset
          .from_tensor_slices(feed)
          .shuffle(buffer_size=1024)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
    return ds


class MNIST:

    def __init__(self, batch_size):
        (lx, ly), (tx, ty) = tf.keras.datasets.mnist.load_data()
        lx, tx = map(lambda tensor: (tensor[..., None] / 255.).astype("float32"), [lx, tx])
        self.batch_size = batch_size
        self.train_steps_per_epoch = math.ceil(len(lx) / batch_size)
        self.test_steps_per_epoch = math.ceil(len(tx) / batch_size)
        self._train = lx, ly
        self._test = tx, ty

    def train_dataset(self, batch_size=None, include_labels=True):
        if batch_size is None:
            batch_size = self.batch_size
        return _make_dataset(self._train, batch_size, include_labels)

    def test_dataset(self, batch_size=None, include_labels=True):
        if batch_size is None:
            batch_size = self.batch_size
        return _make_dataset(self._train, batch_size, include_labels)


def normalize(vector):
    vector_mean = vector.mean()
    vector_std = vector.std()
    result = vector - vector_mean
    if vector_std > 0:
        result /= vector_std
    return result
