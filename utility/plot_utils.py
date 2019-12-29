import numpy as np
from matplotlib.axes import Axes


def plot_line_and_smoothing(x_values, y_values, smoothing_window, axes_obj: Axes):
    half_window = smoothing_window // 2
    smoothing_kernel = np.ones(shape=smoothing_window) / smoothing_window

    axes_obj.plot(x_values, y_values, "r-", alpha=0.5)
    axes_obj.plot(x_values[half_window-1:-half_window], np.convolve(y_values, smoothing_kernel, mode="valid"), "b-")
    axes_obj.grid()
