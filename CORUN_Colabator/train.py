# flake8: noqa
import os.path as osp
from CORUN_Colabator.train_pipeline import train_pipeline

import CORUN_Colabator.archs
import CORUN_Colabator.data
import CORUN_Colabator.models
import CORUN_Colabator.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
