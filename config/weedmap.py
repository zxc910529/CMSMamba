import numpy as np


CLASS_NAMES = [
    "background","sugar beet","weed"
]

COLORMAP = {
    0: (0, 0, 0),       # Background : Black
    1: (0,255,0),       # Sugar beet : Green
    2: (255,0,0),       # Weed : Red
}


RGB_MEAN = [0.3433,0.3507,0.3073]
RGB_STD = [0.0775,0.0767,0.0729]
NIR_MEAN = 0.2758
NIR_STD = 0.0721
RE_MEAN = 0.3192
RE_STD = 0.0830

