"""Visualization utils for debug tool."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


class SliderGroup:
    """Class for grouping a slider with forward and backward buttons."""

    def __init__(
        self, fig, left, bottom, width, height, max_val, step, text, callback=None
    ):
        """Initialize a SliderGroup object.

        Args:
            fig: the plt.figure object.
            left: the left position in the window [0, 1]
            bottom: the position from the bottom of the window [0, 1]
            width: the width of the total group [0, 1]
            height: the height of the total group [0, 1]
            max_val: the maximum val of the slider
            step: The step size of the slider
            text: The text of the slider
            callback: The callback method to execute when the slider changes.
        """
        self._fig = fig
        self._left = left
        self._bottom = bottom
        self._width = width
        self._height = height
        self._max_val = max_val
        self._step = step
        self._text = text
        self._callback = callback

        self._init_widgets()

        if self._callback is not None:
            self._slider.on_changed(self._callback)

    def _init_widgets(self):
        """Create the widgets: the 2 buttons and the slider."""
        l, b, w, h = self._left, self._bottom, self._width, self._height

        button1_ax = plt.axes([l + w / 2 - h, b + h / 2, h, h / 2])
        button2_ax = plt.axes([l + w / 2, b + h / 2, h, h / 2])
        self._slider_ax = plt.axes([l, b, w, h / 2])

        self._button_left = Button(button1_ax, '<')
        self._button_right = Button(button2_ax, '>')
        self._slider = Slider(
            self._slider_ax,
            self._text,
            valmin=0,
            valmax=self._max_val,
            valinit=0,
            valstep=self._step,
        )

        self._button_left.on_clicked(self._backward)
        self._button_right.on_clicked(self._forward)

    def _backward(self, event):
        """Call, when button_left is clicked."""
        if self._slider.val > 0:
            self._slider.set_val(self._slider.val - self._step)

    def _forward(self, event):
        """Call, when button_left is clicked."""
        if self._slider.val < self._max_val:
            self._slider.set_val(self._slider.val + self._step)

    def update_max_val(self, max_val):
        """Update the maximum value for the slider.

        Unfortunatelly one can not update the range, a new slider has to be created.
        """
        # changing the current value of the slider if the original value was larger
        # than the new boundary:
        val = self._slider.val
        if val > max_val:
            val = max_val

        self._slider_ax.cla()

        self._max_val = max_val
        self._slider = Slider(
            self._slider_ax,
            self._text,
            valmin=0,
            valmax=self._max_val,
            valinit=0,
            valstep=self._step,
        )

        self._slider.set_val(val)

        if self._callback is not None:
            self._slider.on_changed(self._callback)

    @property
    def val(self):
        """Return the current value of the slider."""
        return self._slider.val
