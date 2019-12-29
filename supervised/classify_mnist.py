import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from utility import data_utils


@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tf.reduce_mean(loss)


@tf.function
def evaluate(x, y):
    pred = model(x)
    acc = tf.keras.metrics.sparse_categorical_accuracy(y, tf.nn.softmax(pred, axis=-1))
    return tf.reduce_mean(acc)


data = data_utils.MNIST(batch_size=32)

model = tf.keras.models.Sequential([
    tfl.Flatten(input_shape=(28, 28, 1)),
    tfl.Dense(64, activation="relu"),
    tfl.Dense(10)
])
optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

losses = []
accs = []

ds = data.train_dataset()
test_ds = data.test_dataset()

for epoch in range(1, 31):
    print("Epoch", epoch)
    for i, (images, labels) in enumerate(ds, start=1):
        train_loss = train(images, labels)
        train_acc = evaluate(images, labels)
        losses.append(train_loss)
        accs.append(train_acc)
        print(
            f"\rP {i / data.train_steps_per_epoch:>7.2%} - "
            f"Loss {np.mean(losses[-100:]):>7.4f} "
            f"Acc {np.mean(accs[-100:]):>7.2%}",
            end="")

    test_accs = []
    for i, (images, labels) in enumerate(test_ds):
        test_accs.append(evaluate(images, labels))

    print(f" Test acc: {np.mean(test_accs):>7.2%}")
    print()
