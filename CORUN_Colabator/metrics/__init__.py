from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .brisque import calculate_brisque
from .nima import calculate_nima

__all__ = ['calculate_brisque', 'calculate_nima']


def calculate_metric(data, opt):
    """Calculate metric from data and CORUN_Options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
