import numpy as np
import torch

CLASS_NAMES = [
    "background","maize","amaranth","barnyard grass","quickweed","weed other"
]


COLORMAP = {
    0: (0, 0, 0),         # Background : Black
    1: (255, 255, 153),   # Maize : Yellow
    2: (173, 216, 230),   # Amaranth : Light Blue
    3: (221, 160, 221),   # Barnyard grass : Purple
    4: (255, 192, 203),   # Quickweed : Pink
    5: (255, 255, 255),   # Weed other : White
}

RGB_MEAN = [0.4637, 0.4443, 0.4086]
RGB_STD = [0.1234, 0.1153, 0.1127]
NIR_MEAN = 0.4077
NIR_STD = 0.1065
RE_MEAN = 0.4335
RE_STD = 0.1082