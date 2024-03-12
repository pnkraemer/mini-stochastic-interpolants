"""Plotting utilities."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def slider(fun, init_frequency, *, name):
    ts, trajectories = fun(init_frequency)
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    lines = ax.plot(ts, trajectories, linewidth=0.5, color="black", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    freq_slider = Slider(
        ax=axfreq,
        label=name,
        valmin=-3.0,
        valmax=3.0,
        valinit=init_frequency,
        orientation="vertical",
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        _ts, ys_new = fun(freq_slider.val)
        for l, y in zip(lines, ys_new.T):
            l.set_ydata(y)
        fig.canvas.draw_idle()

    # register the update function with each slider
    freq_slider.on_changed(update)
