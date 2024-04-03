from typing import List, Tuple
import colorsys
import random
import numpy as np


def get_colors(num_colors: int, seed=0, without_red=False) -> List[Tuple[int, int, int]]:
    """Gets colormap for points."""
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        if without_red: # replcae red colors, red = (255, 0, 0)
            if color[0] > 200:
                color = (color[0] - 100, color[1], color[2])
        colors.append(color)
    random.seed(seed)
    random.shuffle(colors, )
    return colors
