import cv2
import math
import numpy as np
import os


from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_nima(img_path, nima, **kwargs):
    """Calculate NIMA.

    Args:
        img (str): Input path
        nima (pyiqa.NIMA): NIMA model.

    Returns:
        float: NIMA score.
    """
    # calculate NIMA
    score = nima(img_path)
    # to numpy
    score = score.item()

    return score
