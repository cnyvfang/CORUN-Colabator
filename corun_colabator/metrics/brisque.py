import cv2
import math
import numpy as np
import os


from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_brisque(img, brisque, **kwargs):
    """Calculate BRISQUE.

    Args:
        img (str): Input path
        brisque (pyiqa.BRISQUE): BRISQUE model.
    Returns:
        float: BRISQUE score.
    """

    # calculate BRISQUE
    score = brisque(img)
    # to numpy
    score = score.item()

    return score
