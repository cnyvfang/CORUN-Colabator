import importlib
from basicsr.utils import scandir
from os import path as osp

from copy import deepcopy
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'DASUM.archs.{file_name}') for file_name in arch_filenames]



def build_network(opt, kwargs=None):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    if kwargs is None:
        net = ARCH_REGISTRY.get(network_type)(**opt)
    else:
        net = ARCH_REGISTRY.get(network_type)(**opt, **kwargs)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net