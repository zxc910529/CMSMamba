import numpy as np

CLASS_NAMES = [
    "background","crop","weed"
]

COLORMAP = {
    0: (40, 40, 40),      # Background : Gray
    1: (100, 150, 80),   # Crop : Green
    2: (130, 100, 70),   # Weed : Brown
}

RGB_MEAN = [0.5875,0.5377,0.3991]
RGB_STD = [0.1558,0.1280,0.1010]
NIR_MEAN = 0.5042
NIR_STD = 0.1845
