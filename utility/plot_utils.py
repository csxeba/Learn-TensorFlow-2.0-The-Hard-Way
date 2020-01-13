import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt


def plot_line_and_smoothing(x_values, y_values, smoothing_window, axes_obj: Axes):
    half_window = smoothing_window // 2
    smoothing_kernel = np.ones(shape=smoothing_window) / smoothing_window

    axes_obj.plot(x_values, y_values, "r-", alpha=0.5)
    axes_obj.plot(x_values[half_window-1:-half_window], np.convolve(y_values, smoothing_kernel, mode="valid"), "b-")
    axes_obj.grid()


def plot_vectors(vectors, names, smoothing_window_size, skip_first=10, show=True, **subplots_kwargs):
    figsize = subplots_kwargs.pop("figsize", (16, 9))
    fig, axes = plt.subplots(len(vectors), sharex="all", figsize=figsize, **subplots_kwargs)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    half_window = smoothing_window_size // 2

    for ax, vec, name in zip(axes, vectors, names):
        vec = vec[skip_first:]
        ax.plot(vec[half_window:-half_window], "r-", alpha=0.5)
        ax.plot(np.convolve(vec, np.ones(smoothing_window_size) / smoothing_window_size, mode="valid"))
        ax.set_title(name)
        ax.grid()

    if show:
        plt.tight_layout()
        plt.show()


def plot_history(history, smoothing_window_size, skip_first=10, show=True, **subplots_kwargs):
    vectors = [v for k, v in history.gather().items()]
    plot_vectors(vectors, history.keys, smoothing_window_size, skip_first, show=show, **subplots_kwargs)
