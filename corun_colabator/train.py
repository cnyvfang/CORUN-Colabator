# flake8: noqa
import os.path as osp
from corun_colabator.train_pipeline import train_pipeline

import corun_colabator.archs
import corun_colabator.data
import corun_colabator.models
import corun_colabator.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
